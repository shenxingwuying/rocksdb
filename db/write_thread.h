//  Copyright (c) 2013, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.

#pragma once

#include <assert.h>
#include <stdint.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <type_traits>
#include "db/flush_scheduler.h"
#include "db/write_batch_internal.h"
#include "rocksdb/status.h"
#include "util/autovector.h"
#include "util/instrumented_mutex.h"

namespace rocksdb {

class WriteThread {
 public:
  enum State : uint8_t {
    // The initial state of a writer.  This is a Writer that is
    // waiting in JoinBatchGroup.  This state can be left when another
    // thread informs the waiter that it has become a group leader
    // (-> STATE_GROUP_LEADER), when a leader that has chosen to be
    // non-parallel informs a follower that its writes have been committed
    // (-> STATE_COMMITTED), or when a leader that has chosen to perform
    // updates in parallel and needs this Writer to apply its batch (->
    // STATE_PARALLEL_FOLLOWER).
    STATE_INIT,

    // The state used to inform a waiting Writer that it has become the
    // leader, and it should now build a write batch group.  Tricky:
    // this state is not used if newest_writer_ is empty when a writer
    // enqueues itself, because there is no need to wait (or even to
    // create the mutex and condvar used to wait) in that case.  This is
    // a terminal state.
    STATE_GROUP_LEADER,

    // A Writer that has returned as a follower in a parallel group.
    // It should apply its batch to the memtable and then enter
    // STATE_AWAITING_COMMIT by calling CompleteParallelFollower.
    STATE_PARALLEL_FOLLOWER,

    // A Writer that is waiting for the leader to finish the in-memory
    // commit after a concurrent write.
    STATE_AWAITING_PARALLEL_COMMIT,

    // A follower whose writes have been applied.  This is a terminal state.
    STATE_COMMITTED,
  };

  // Information kept for every waiting writer.
  struct Writer {
    WriteBatch* batch;
    bool sync;
    bool disableWAL;
    bool in_batch_group;
    bool has_callback;
    bool made_waitable;       // records lazy construction of mutex and cv
    uint8_t state;            // read/write under StateMutex() (or pre-link)
    SequenceNumber sequence;  // the sequence number to use
    std::vector<ColumnFamilyData*> cfd_set;  // used only for concurrent adds
    Status status;
    std::aligned_storage<sizeof(std::mutex)>::type state_mutex_bytes;
    std::aligned_storage<sizeof(std::condition_variable)>::type state_cv_bytes;
    Writer* link_older;  // read/write only before linking, or as leader
    Writer* link_newer;  // lazy, read/write only before linking, or as leader

    Writer()
        : batch(nullptr),
          sync(false),
          disableWAL(false),
          in_batch_group(false),
          has_callback(false),
          made_waitable(false),
          state(STATE_INIT),
          link_older(nullptr),
          link_newer(nullptr) {}

    ~Writer() {
      if (made_waitable) {
        StateMutex().~mutex();
        StateCV().~condition_variable();
      }
    }

    void CreateMutex() {
      assert(state == STATE_INIT);
      if (!made_waitable) {
        // Note that made_waitable is tracked separately from state
        // transitions, because we can't atomically create the mutex and
        // link into the list.
        made_waitable = true;
        new (&state_mutex_bytes) std::mutex;
        new (&state_cv_bytes) std::condition_variable;
      }
    }

    // No other mutexes may be acquired while holding StateMutex(), it is
    // always last in the order
    std::mutex& StateMutex() {
      assert(made_waitable);
      return *static_cast<std::mutex*>(static_cast<void*>(&state_mutex_bytes));
    }

    std::condition_variable& StateCV() {
      assert(made_waitable);
      return *static_cast<std::condition_variable*>(
                 static_cast<void*>(&state_cv_bytes));
    }
  };

  WriteThread() : newest_writer_(nullptr) {}

  // IMPORTANT: None of the methods in this class rely on the db mutex
  // for correctness. All of the methods except JoinBatchGroup and
  // EnterUnbatched may be called either with or without the db mutex held.
  // Correctness is maintained by ensuring that only a single thread is
  // a leader at a time.

  // Registers w as ready to become part of a batch group, waits until the
  // caller should perform some work, and returns the current state of the
  // writer.  If w has become the leader of a write batch group, returns
  // STATE_GROUP_LEADER.  If w has been made part of a sequential batch
  // group and the leader has performed the write, returns STATE_DONE.
  // If w has been made part of a parallel batch group and is reponsible
  // for updating the memtable, returns STATE_PARALLEL_FOLLOWER.
  //
  // The db mutex SHOULD NOT be held when calling this function, because
  // it will block.
  //
  // Writer* w:        Writer to be executed as part of a batch group
  void JoinBatchGroup(Writer* w);

  // Reports the completion of w's batch to the parallel group leader,
  // and waits for the rest of the parallel batch to complete.
  void CompleteParallelFollower(Writer* w);

  // Constructs a write batch group led by leader, which should be a
  // Writer passed to JoinBatchGroup on the current thread.
  //
  // Writer* leader:         Writer that is STATE_GROUP_LEADER
  // Writer** last_writer:   Out-param that identifies the last follower
  // autovector<WriteBatch*>* write_batch_group: Out-param of group members
  // returns:                Total batch group byte size
  size_t EnterAsBatchGroupLeader(Writer* leader, Writer** last_writer,
                                 autovector<WriteBatch*>* write_batch_group);

  // Causes JoinBatchGroup to return STATE_PARALLEL_FOLLOWER for all of the
  // non-leader members of this write batch group.  Sets Writer::sequence
  // before waking them up.
  //
  // Writer* leader:          Writer passed to JoinBatchGroup, but !done
  // Writer* last_writer:     Value of out-param of EnterAsBatchGroupLeader
  // SequenceNumber sequence: Starting sequence number to assign to Writer-s
  void LaunchParallelFollowers(Writer* leader, Writer* last_writer,
                               SequenceNumber sequence);

  // Waits until all parallel followers have called
  // CompleteParallelFollower.  Should only be called by the leader,
  // and only if LaunchParallelFollowers was called alread.
  //
  // Writer* leader:         From EnterAsBatchGroupLeader
  // Writer* last_writer:    Value of out-param of EnterAsBatchGroupLeader
  // returns:                Joined status from leader and all followers
  Status JoinParallelFollowers(Writer* leader, Writer* last_writer);

  // Performs the delayed CheckMemtableFull work that was accumulated
  // during parallel writes.
  //
  // Writer* leader:         From EnterAsBatchGroupLeader
  // Writer* last_writer:    Value of out-param of EnterAsBatchGroupLeader
  // FlushScheduler* flush_scheduler:  FlushSchedhuler if memtable is full
  void CheckMemtableFull(Writer* leader, Writer* last_writer,
                         FlushScheduler* flush_scheduler);

  // Unlinks the Writer-s in a batch group, wakes up the non-leaders, and
  // wakes up the next leader (if any).
  //
  // Writer* leader:         From EnterAsBatchGroupLeader
  // Writer* last_writer:    Value of out-param of EnterAsBatchGroupLeader
  // Status status:          Status of write operation
  void ExitAsBatchGroupLeader(Writer* leader, Writer* last_writer,
                              Status status);

  // Waits for all preceding writers (unlocking mu while waiting), then
  // registers w as the currently proceeding writer.
  //
  // Writer* w:              A Writer not eligible for batching
  // InstrumentedMutex* mu:  The db mutex, to unlock while waiting
  // REQUIRES: db mutex held
  void EnterUnbatched(Writer* w, InstrumentedMutex* mu);

  // Completes a Writer begun with EnterUnbatched, unblocking subsequent
  // writers.
  void ExitUnbatched(Writer* w);

 private:
  // Points to the newest pending Writer.  Only leader can remove
  // elements, adding can be done lock-free by anybody
  std::atomic<Writer*> newest_writer_;

  void AwaitStateExit(Writer* w, uint8_t state_to_exit);
  void AwaitCommitted(Writer* w);
  void SetState(Writer* w, uint8_t new_state);

  // Links w into the newest_writer_ list. Sets *linked_as_leader to
  // true if w was linked directly into the leader position.  Safe to
  // call from multiple threads without external locking.
  void LinkOne(Writer* w, bool* linked_as_leader);

  // Computes any missing link_newer links.  Should not be called
  // concurrently with itself.
  void CreateMissingNewerLinks(Writer* head);
};

}  // namespace rocksdb
