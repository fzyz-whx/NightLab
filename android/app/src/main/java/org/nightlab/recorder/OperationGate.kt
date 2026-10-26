package org.nightlab.recorder

import java.util.concurrent.atomic.AtomicReference

/**
 * Process-wide reservation held across queued work and Activity recreation.
 * Reserve before queueing/starting work and release exactly once in its final cleanup.
 * Keep one shared instance; callers must also track their own operation generation,
 * since an owner enum alone is not a token identifying a particular acquisition.
 */
class OperationGate {
    enum class Owner { IDLE, CAPTURE, RECOVERY, EXPORT, DELETE }

    private val current = AtomicReference(Owner.IDLE)

    val owner: Owner
        get() = current.get()

    fun tryAcquire(owner: Owner): Boolean {
        require(owner != Owner.IDLE) { "空闲状态不能申请操作权限" }
        return current.compareAndSet(Owner.IDLE, owner)
    }

    fun release(owner: Owner) {
        require(owner != Owner.IDLE) { "空闲状态不能释放操作权限" }
        check(current.compareAndSet(owner, Owner.IDLE)) { "操作权限不属于 $owner，不能释放其他操作" }
    }
}
