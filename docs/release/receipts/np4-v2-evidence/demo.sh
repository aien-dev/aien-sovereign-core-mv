# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

V=$1; P=$2
BIN=$V/bin; CFG=$V/config; SA=$V/standalone/install.sh
inst(){ rel=$1; shift; env -i PATH=/usr/bin:/bin HOME=$V/home AIEN_RELEASE_TAG=v0.0.0-np4 AIEN_RELEASE_BASE_URL=file://$V/$rel AIEN_BIN_DIR=$BIN AIEN_CONFIG_DIR=$CFG AIEN_INSTALL_NO_PROFILE=1 AIEN_ALLOWED_SIGNERS=$P/keys/allowed_signers "$@" ; }
run_aien(){ (cd $V/empty && env -i PATH=$V/empty HOME=$V/home $BIN/aien --version); }
live(){ readlink $CFG/releases/current; }
echo "## 1 INSTALL"; inst relA bash $SA 2>&1 | grep -v '^ *aien \|^  *Quick\|====' | sed -n 4,30p
sha256sum $(readlink -f $BIN/aien); ls -l $BIN | sed 's/.* \([^ ]* -> .*\)/\1/' | head -9; echo "live=$(live)"; cat $CFG/installed.toml
echo "## 2 EXECUTE"; run_aien; bash $P/build2/scripts/zero-cuda-gate.sh $BIN/aien | cut -c1-70
echo "# user edit" >> $CFG/operator.toml; sha256sum $CFG/operator.toml > $V/op.sum
echo "## 3 UPGRADE (candidate equal CAND-3, harmless version string change)"; inst relB bash $SA 2>&1 | grep -n 'Live release\|Verifying\|Error'
sha256sum $(readlink -f $BIN/aien); run_aien; echo "live=$(live)"; cat $CFG/installed.toml; ls $CFG/releases; (cd $V; sha256sum -c op.sum)
echo "## 3b DOWNGRADE GUARD (repackaged as CAND-2)"; inst relOld bash $SA 2>&1 | grep 'Error'; echo "live still $(live)"
echo "## 4 ROLLBACK"; env -i PATH=/usr/bin:/bin HOME=$V/home AIEN_BIN_DIR=$BIN AIEN_CONFIG_DIR=$CFG bash $SA --rollback 2>&1 | grep -v '====\|⚡'
sha256sum $(readlink -f $BIN/aien); run_aien; echo "live=$(live)"; grep '^current\|^previous\|^last-action' $CFG/installed.toml; (cd $V; sha256sum -c op.sum)
echo "## 5 INTERRUPTED UPGRADE to a fresh third release relC (kill -9 at three points, then rerun)"
for point in mid-copy after-copy before-swap; do
  OLD=$(live); OLDSHA=$(sha256sum $(readlink -f $BIN/aien)|cut -c1-12)
  inst relC env AIEN_TEST_PAUSE_AT=$point bash $SA > $V/int.log 2>&1 &
  pid=$!; for i in $(seq 1 100); do grep -q "paused at $point" $V/int.log && break; sleep 0.1; done
  grep -q "paused at $point" $V/int.log || echo "NEVER PAUSED"
  pkill -9 -P $pid; kill -9 $pid; wait $pid 2>/dev/null
  echo "killed at $point: live=$(live) (was $OLD) aien=$(sha256sum $(readlink -f $BIN/aien)|cut -c1-12) (was $OLDSHA) runs: $(run_aien) | leftovers: $(ls -A $CFG/releases | tr '\n' ' ')"
done
echo "rerun after interruption (relC):"; inst relC bash $SA 2>&1 | grep 'Live release\|Error'; echo "live=$(live) aien=$(sha256sum $(readlink -f $BIN/aien)|cut -c1-12)"; run_aien; ls -A $CFG/releases
echo "## 6 pinned key refusal (no signer override)"; env -i PATH=/usr/bin:/bin HOME=$V/home AIEN_RELEASE_TAG=v0.0.0-np4 AIEN_RELEASE_BASE_URL=file://$V/relA AIEN_BIN_DIR=$V/binX AIEN_CONFIG_DIR=$V/cfgX AIEN_INSTALL_NO_PROFILE=1 bash $SA 2>&1 | tail -1; ls $V/binX 2>&1 | head -1
