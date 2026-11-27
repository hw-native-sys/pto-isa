# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check the installer preserves Git TLS configuration without accessing the network."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


INSTALLER = Path(__file__).resolve().parents[2] / "scripts" / "install_pto.sh"
BASH = shutil.which("bash")
GIT = shutil.which("git")


@unittest.skipUnless(BASH and GIT and shutil.which("mkdir"), "Bash, Git, and mkdir are required")
class TestInstallerGitConfig(unittest.TestCase):
    def check_setting(self, initial):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            home = root / "home"
            home.mkdir()
            bin_dir = root / "bin"
            bin_dir.mkdir()
            global_config = home / ".gitconfig"
            git_log = root / "git-commands.log"
            env = os.environ.copy()
            env.update(
                HOME=str(home),
                USERPROFILE=str(home),
                XDG_CONFIG_HOME=str(home / ".config"),
                GIT_CONFIG_GLOBAL=str(global_config),
                GIT_CONFIG_NOSYSTEM="1",
                TEST_REAL_GIT=Path(GIT).as_posix(),
                TEST_GIT_LOG=git_log.as_posix(),
                PATH=str(bin_dir) + os.pathsep + env.get("PATH", ""),
            )
            env.pop("GIT_CONFIG_COUNT", None)
            if initial is not None:
                subprocess.run([GIT, "config", "--global", "http.sslVerify", initial], env=env, cwd=root, check=True)
            before = global_config.read_bytes() if global_config.exists() else None
            commands = {
                "git": """#!/bin/sh
printf '%s\\n' "$*" >> "$TEST_GIT_LOG"
case "$1" in
  config) exec "$TEST_REAL_GIT" "$@" ;;
  clone) mkdir -p googletest ;;
  *) exit 99 ;;
esac
""",
                "cmake": "#!/bin/sh\nexit 0\n",
                "make": "#!/bin/sh\nexit 0\n",
                "sudo": "#!/bin/sh\nexit 0\n",
                "nproc": "#!/bin/sh\nprintf '1\\n'\n",
            }
            for name, content in commands.items():
                command = bin_dir / name
                with command.open("w", encoding="utf-8", newline="\n") as stream:
                    stream.write(content)
                command.chmod(0o755)
            source = INSTALLER.read_text(encoding="utf-8")
            start = source.index("install_gtest() {")
            end = source.index("\nmain() {", start)
            # Exercise the production function while preventing downloads, compilation, and installation.
            driver = "info() { :; }\nerror() { :; }\n" + source[start:end] + "\ninstall_gtest\n"
            result = subprocess.run(
                [BASH, "-s"],
                input=driver,
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            after = global_config.read_bytes() if global_config.exists() else None
            self.assertEqual(after, before, "The installer modified the user's global Git configuration")
            self.assertEqual(
                git_log.read_text().splitlines(), ["clone https://github.com/google/googletest.git -b v1.14.x"]
            )

    def test_preserves_enabled_certificate_verification(self):
        self.check_setting("true")

    def test_preserves_disabled_certificate_verification(self):
        self.check_setting("false")

    def test_does_not_add_certificate_verification_override(self):
        self.check_setting(None)


if __name__ == "__main__":
    unittest.main()
