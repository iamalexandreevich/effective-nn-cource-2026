# Как подключить GPTQ-модель к общему замеру

## Собрать чекпоинт

Команды ниже выполняются из каталога `whisper-quantization`. Нужны Python 3.11, CMake, компилятор C++ и FFmpeg.

```sh
python3.11 -m venv .runtime/venv
.runtime/venv/bin/python -m pip install -r requirements.txt
git clone --depth 1 --branch v1.8.3 https://github.com/ggml-org/whisper.cpp.git .runtime/whisper.cpp
cmake -S .runtime/whisper.cpp -B .runtime/build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DWHISPER_COREML=OFF -DWHISPER_BUILD_SERVER=OFF -DWHISPER_BUILD_TESTS=OFF
cmake --build .runtime/build --target whisper-cli whisper-quantize -j 8

mkdir -p .runtime/models
curl -fL https://huggingface.co/ggerganov/whisper.cpp/resolve/80da2d8/ggml-small.bin \
  -o .runtime/models/ggml-small.bin
shasum -a 256 .runtime/models/ggml-small.bin
.runtime/build/bin/whisper-quantize .runtime/models/ggml-small.bin .runtime/models/small-q4_0.bin q4_0

.runtime/venv/bin/python fetch_asr.py .runtime/data/libri-cal --split train.100 \
  --offsets 0 3600 7200 10800 14400 18000 21600 25200
VECLIB_MAXIMUM_THREADS=4 .runtime/venv/bin/python calibrate.py \
  --f16 .runtime/models/ggml-small.bin --control .runtime/models/small-q4_0.bin \
  --manifest .runtime/data/libri-cal/manifest.jsonl --counts 16 64 \
  --output-dir .runtime/models/full
```

Ожидаемый commit whisper.cpp — `2eeeba56e9edd762b4b38467bab96c2517163158`. SHA-256 исходного F16 — `1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b`. На машине замера сборка потребовала SDK macOS 26.5 через `-DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`. После сборки возьмите фактический SHA-256 из JSON рядом с моделью. [Хеши и калибровочные IDs пилота](../results/calibration.json)

## Подключить к командному бенчмарку

Скопируйте готовый `gptq-q4_0-16.bin` под именем `.work/models/team_gptq.bin` в рабочую папку `EDLM_Whisper_Boost`. Из корня командного репозитория после его настройки:

```sh
shasum -a 256 .work/models/team_gptq.bin
uv run research/benchmark.py smoke team_gptq+ctx768 --backend cpu --repeats 1 --output .work/team-gptq-smoke
uv run research/benchmark.py m3_panel q4_0+ctx768 team_gptq+ctx768 \
  --backend cpu --threads 4 --repeats 1 --output .work/team-gptq-panel/raw
uv run research/analyze.py m3_panel --raw .work/team-gptq-panel/raw \
  --out .work/team-gptq-panel --summary-only
```

Для оценки GPTQ берите тот же Q4_0-контроль, окно, runtime, устройство и декодирование. Для оценки окна меняйте только окно и явно указывайте состояние FA. Наши чекпоинты прошли проверку в штатном whisper.cpp v1.8.3. Команды выше используют интерфейс командного репозитория. [Командная инструкция](https://github.com/mnenadoeloo/EDLM_Whisper_Boost/blob/579249a889931b33a0514396dab484fa1ce8465f/docs/team-merge.md)

Если команда собирает смешанный checkpoint из F16, `gptq(weights, second_moment, damp).pack()` возвращает байты одного тензора Q4_0. `weights` имеет форму `[out_features, in_features]`. Строки и столбцы второго момента должны идти в порядке входных каналов исходного тензора. Применяйте этот метод к Q4_0-тензорам и экспортируйте каждый тензор напрямую из F16. [Реализация GPTQ](../q4.py)

## Повтор прогона с аудиоокном

Соберите предыдущий штатный Q4_0. Подготовьте старые dev-манифесты для исключения их записей, затем получите Parquet фиксированных ревизий.

```sh
.runtime/venv/bin/python -m pip install pyarrow
.runtime/venv/bin/python fetch_asr.py .runtime/data/libri-dev --split validation --per-offset 4 \
  --offsets 0 320 640 960 1280 1600 1920 2240
.runtime/venv/bin/python fetch_asr.py .runtime/data/golos-dev \
  --dataset bond005/sberdevices_golos_100h_farfield --config default --split validation \
  --per-offset 4 --offsets 0 50 100 150 200 250 300 350

mkdir -p .runtime/context-study
curl -fL https://huggingface.co/datasets/openslr/librispeech_asr/resolve/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1/clean/validation/0000.parquet \
  -o .runtime/context-study/libri-dev.parquet
curl -fL https://huggingface.co/datasets/bond005/sberdevices_golos_100h_farfield/resolve/c93949f7140beef4adc404e7b54841e957f81c54/data/validation-00000-of-00001-fd7b65d8512dcdb5.parquet \
  -o .runtime/context-study/golos-dev.parquet

.runtime/venv/bin/python experiments/context-study/prepare_sample.py libri
.runtime/venv/bin/python experiments/context-study/prepare_sample.py golos
.runtime/venv/bin/python experiments/context-study/build_timed_cli.py
.runtime/venv/bin/python experiments/context-study/run_study.py
PYTHONPATH=. .runtime/venv/bin/python experiments/context-study/summarize_study.py
```

Сборка измерительного CLI использует файлы `flags.make` и `link.txt` генератора CMake Unix Makefiles и добавляет только замер времени вызова распознавания. Скрипты сохраняют команды, предсказания и тайминги под `.runtime/context-study/`. [Отчёт](../results/REPORT.md) · [Сводные метрики](../results/context-study.json)
