import shlex
import subprocess
from pathlib import Path

root = Path(".runtime").resolve()
work = root / "context-study"
cli = root / "whisper.cpp/examples/cli/cli.cpp"
original = cli.read_text()
anchor = "            if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {"
assert original.count(anchor) == 1
timed = """            const auto study_started = std::chrono::steady_clock::now();
            const auto study_status = whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors);
            const auto study_elapsed = std::chrono::steady_clock::now() - study_started;
            fprintf(stderr, "\\nSTUDY_CLIP_MICROS %d %lld\\n", f,
                    static_cast<long long>(std::chrono::duration_cast<std::chrono::microseconds>(study_elapsed).count()));
            if (study_status != 0) {"""
instrumented = work / "cli_timed.cpp"
instrumented.write_text("#include <chrono>\n" + original.replace(anchor, timed))
build = root / "build/examples/cli"
cmake_target = build / "CMakeFiles/whisper-cli.dir"
make_flags = (cmake_target / "flags.make").read_text().splitlines()
flags = [flag for line in make_flags if line.startswith(("CXX_DEFINES =", "CXX_INCLUDES =", "CXX_FLAGS ="))
         for flag in shlex.split(line.partition("=")[2])]
command = shlex.split((cmake_target / "link.txt").read_text())
object_path = work / "cli_timed.o"
subprocess.run([command[0], *flags, "-c", str(instrumented), "-o", str(object_path)], cwd=build, check=True)
command = [str(object_path) if value.endswith("cli.cpp.o") else value for value in command]
command[command.index("-o") + 1] = str(work / "whisper-cli-timed")
subprocess.run(command, cwd=build, check=True)
