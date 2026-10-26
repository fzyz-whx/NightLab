package org.nightlab.recorder

import java.util.concurrent.CountDownLatch
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.junit.Assert.*
import org.junit.Test
import org.nightlab.recorder.OperationGate.Owner

class OperationGateTest {
    private inline fun <reified T : Throwable> refuses(block: () -> Unit) {
        try { block() } catch (error: Throwable) {
            if (error is T) return
            throw AssertionError("Expected ${T::class.java.name}, got $error", error)
        }
        fail("Expected ${T::class.java.name}")
    }

    @Test fun transitionsRequireAnActualOperationAndMatchingRelease() {
        val gate = OperationGate()
        assertEquals(Owner.IDLE, gate.owner)
        refuses<IllegalArgumentException> { gate.tryAcquire(Owner.IDLE) }
        refuses<IllegalArgumentException> { gate.release(Owner.IDLE) }
        refuses<IllegalStateException> { gate.release(Owner.CAPTURE) }
        assertTrue(gate.tryAcquire(Owner.CAPTURE))
        assertFalse(gate.tryAcquire(Owner.CAPTURE))
        assertEquals(Owner.CAPTURE, gate.owner)
        gate.release(Owner.CAPTURE)
        assertEquals(Owner.IDLE, gate.owner)
        refuses<IllegalStateException> { gate.release(Owner.CAPTURE) }
    }

    @Test fun simultaneousContendersHaveExactlyOneOwner() {
        val gate = OperationGate()
        val ready = CountDownLatch(2)
        val start = CountDownLatch(1)
        val won = AtomicInteger()
        val executor = Executors.newFixedThreadPool(2)
        try {
            val attempts = listOf(Owner.CAPTURE, Owner.RECOVERY).map { owner ->
                executor.submit<Boolean> {
                    ready.countDown()
                    check(start.await(5, TimeUnit.SECONDS))
                    gate.tryAcquire(owner).also { if (it) won.incrementAndGet() }
                }
            }
            assertTrue(ready.await(5, TimeUnit.SECONDS))
            start.countDown()
            assertEquals(1, attempts.count { it.get(5, TimeUnit.SECONDS) })
            assertEquals(1, won.get())
            assertNotEquals(Owner.IDLE, gate.owner)
            gate.release(gate.owner)
            assertEquals(Owner.IDLE, gate.owner)
        } finally {
            start.countDown()
            executor.shutdownNow()
        }
    }

    @Test fun queuedRecoveryPreventsCaptureBeforeWorkerStarts() {
        val gate = OperationGate()
        val workerMayStart = CountDownLatch(1)
        val executor = Executors.newSingleThreadExecutor()
        assertTrue(gate.tryAcquire(Owner.RECOVERY))
        try {
            val task = executor.submit {
                try { check(workerMayStart.await(5, TimeUnit.SECONDS)) }
                finally { gate.release(Owner.RECOVERY) }
            }
            // Reservation already applies even though the recovery worker has not run.
            assertFalse(gate.tryAcquire(Owner.CAPTURE))
            assertFalse(gate.tryAcquire(Owner.EXPORT))
            workerMayStart.countDown()
            task.get(5, TimeUnit.SECONDS)
            assertTrue(gate.tryAcquire(Owner.CAPTURE))
            gate.release(Owner.CAPTURE)
        } finally {
            workerMayStart.countDown()
            executor.shutdownNow()
        }
    }

    @Test fun captureReservationProtectsSessionCreationUntilFinalCleanup() {
        val gate = OperationGate()
        val sessionCreated = CountDownLatch(1)
        val captureMayFinish = CountDownLatch(1)
        val executor = Executors.newSingleThreadExecutor()
        assertTrue(gate.tryAcquire(Owner.CAPTURE))
        try {
            val task = executor.submit {
                try {
                    // Models the interval after createSession but before live ID publication.
                    sessionCreated.countDown()
                    check(captureMayFinish.await(5, TimeUnit.SECONDS))
                } finally { gate.release(Owner.CAPTURE) }
            }
            assertTrue(sessionCreated.await(5, TimeUnit.SECONDS))
            assertFalse(gate.tryAcquire(Owner.RECOVERY))
            assertFalse(gate.tryAcquire(Owner.DELETE))
            captureMayFinish.countDown()
            task.get(5, TimeUnit.SECONDS)
            assertTrue(gate.tryAcquire(Owner.RECOVERY))
            gate.release(Owner.RECOVERY)
        } finally {
            captureMayFinish.countDown()
            executor.shutdownNow()
        }
    }

    @Test fun replacementActivitySeesExistingExportReservation() {
        val processGate = OperationGate()
        val firstActivityGate = processGate
        assertTrue(firstActivityGate.tryAcquire(Owner.EXPORT))
        // Recreated UI receives the same process singleton, not a fresh lock.
        val replacementActivityGate = processGate
        assertEquals(Owner.EXPORT, replacementActivityGate.owner)
        assertFalse(replacementActivityGate.tryAcquire(Owner.CAPTURE))
        assertFalse(replacementActivityGate.tryAcquire(Owner.RECOVERY))
        assertFalse(replacementActivityGate.tryAcquire(Owner.DELETE))
        firstActivityGate.release(Owner.EXPORT)
        assertTrue(replacementActivityGate.tryAcquire(Owner.CAPTURE))
        replacementActivityGate.release(Owner.CAPTURE)
    }

    @Test fun wrongReleaseCannotDropAnotherOperationsReservation() {
        val gate = OperationGate()
        for (owner in listOf(Owner.CAPTURE, Owner.RECOVERY, Owner.EXPORT, Owner.DELETE)) {
            assertTrue(gate.tryAcquire(owner))
            for (wrong in listOf(Owner.CAPTURE, Owner.RECOVERY, Owner.EXPORT, Owner.DELETE).filter { it != owner }) {
                refuses<IllegalStateException> { gate.release(wrong) }
                assertEquals(owner, gate.owner)
                assertFalse(gate.tryAcquire(wrong))
            }
            gate.release(owner)
        }
    }
}
