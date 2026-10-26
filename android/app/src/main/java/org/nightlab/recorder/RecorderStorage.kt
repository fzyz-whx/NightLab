package org.nightlab.recorder

import android.content.Context
import java.io.File

/** Both service and UI workers share one lock-protected store within this process. */
object RecorderStorage {
    val operations = OperationGate()
    private var instance: SessionStore? = null
    @Synchronized fun get(context: Context): SessionStore = instance ?: SessionStore(
        File(context.applicationContext.filesDir, "recordings")
    ).also { instance = it }
}
