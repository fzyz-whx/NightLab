# SleepAccel31: bind a fresh machine's own intake receipt

The public 31-person manifest records the original verified intake receipt's
SHA256. The receipt itself remains private: it includes the intake machine's
absolute archive path and elapsed time. A real intake on another machine creates
different receipt bytes even when all 93 official raw files are identical.

Use this procedure **only in a fresh checkout, before preparing the 31-person
cohort or creating model runs**. Keep any existing experiment checkout frozen.
The binding command rejects a changed receipt when the cohort processed directory
contains files, or when either `models/` or `runs/` is populated. The latter guard
is deliberately conservative because experiments can use custom directory names.
An already matching verified receipt is a read-only no-change operation even in
an existing experiment checkout.

1. Follow the approved archive acquisition instructions in
   [INSTALLATION.md](INSTALLATION.md). Obtain the exact approved 1.0.0 ZIP and the
   official checksum/license evidence locally. Acquisition authorization remains
   limited to that archive; this command performs no downloads.
2. Actually run the archive intake on this machine. From the repository root,
   using the installed project Python environment:

   ```powershell
   python -m scripts.datasets.sleepaccel_archive --execute
   ```

   Intake must finish successfully and write
   `data/raw/sleepaccel/archives/intake-receipt.json`. Do not copy another
   machine's receipt, edit a receipt to imitate success, or copy private history.
3. Validate the binding with its default read-only mode:

   ```powershell
   python -m scripts.datasets.bind_sleepaccel31_receipt
   ```

   This streams all 93 actual raw files to verify official SHA256 and exact size.
   It also verifies the approved source plan/checksum evidence, receipt schema,
   archive identity, member mapping, intake source and immutable manifest metadata.
   It uses one sequential worker with no numerical runtime or GPU. A missing,
   incomplete or inconsistent receipt is an error; there is no bypass flag.
4. In that fresh checkout, explicitly bind the verified local receipt:

   ```powershell
   python -m scripts.datasets.bind_sleepaccel31_receipt --execute
   git diff -- data/manifests/sleepaccel-cohort-31.json
   ```

   The only changed field is `acquisition.intake_receipt_sha256`. Review and, when
   saving this local reproduction state, commit that one metadata change in your
   own checkout. The ignored `private/sleepaccel31-receipt-binding.json` records
   the original public receipt digest, the prior/current local receipt digests and
   before/after manifest hashes. It contains no receipt copy or absolute path.
   Keep that provenance record private along with the actual intake receipt.
5. Continue with the documented cohort audit/preparation and reproduction steps.
   The existing auditor now checks this machine's actual receipt. The existing
   preparation source hashes record the locally bound manifest and receipt.

The approved archive remains exactly **576850847 bytes**, with local SHA256
`f0c1345070823a21e2bcc23a0a2f0cd82b85add5fc7797713dd7ab64a224ff8a`.
This digest is a local archive identity, not an official archive checksum. The
official individual-file SHA256 values, 31 subject IDs, 18/6/7 split, label
mapping, source plan and `sleepaccel-canonical-v3` preparation policy remain
fixed. The binder additionally pins all public manifest metadata, with only the
receipt hash masked, to prevent incidental changes from entering this operation.

The local manifest and receipt provenance hashes consequently differ from the
original experiment checkout. Preparation still uses the same v3 feature/cache
versions and scientific protocol; this is an acquisition provenance change, not
a scientific score change or evidence that retrained scores match. Generate new
local caches and report measured reproduction results honestly. Never rebind an
existing trained checkout to make its old caches appear current.

Before preparation, Git can restore the public manifest if you abandon this
reproduction setup. Once caches/models/runs exist, keep the manifest associated
with those artifacts; use a separate fresh checkout for another intake rather
than rewriting provenance. No original raw files, existing caches or trained
results are modified by binding.
