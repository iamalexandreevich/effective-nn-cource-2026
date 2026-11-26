# GPTQ для Whisper Small в whisper.cpp

Эксперимент Александра Иванова. Проверяем, меняет ли GPTQ качество распознавания при том же формате и размере модели. Основные корпуса команды — LibriSpeech и Golos Farfield. Результаты по GPTQ предназначены для общей презентации вместе с материалами по mixed precision и weighted quantization.


[Отчёт для команды](results/REPORT.md) · [Презентация](results/whisper-small-gptq.pptx) · [Как подключить модель](docs/team-merge.md)

## Что сравниваем

| Модель | Encoder | Decoder |
| --- | --- | --- |
| F16 | Исходные веса Whisper Small | Исходные веса Whisper Small |
| Штатный Q4_0 | Штатная квантизация | Штатная квантизация |
| GPTQ Q4_0 | 72 линейных тензора обработаны GPTQ. Остальные совпадают со штатным Q4_0 | Побайтно совпадает со штатным Q4_0 |

Encoder для калибровки загружается из того же GGML F16 в OpenAI Whisper. Входы линейных слоёв собираются в float32 до квантизации. Калибровка использует первые 30 секунд каждой записи и исключает из статистик кадры за пределами фактического аудио. Вторые моменты усредняются по кадрам. GPTQ использует 1% диагонального демпфирования и обрабатывает каналы в исходном порядке. Каждая непрерывная группа из 32 весов в строке получает signed scale по текущим скорректированным весам. Scale округляется до FP16 до выбора кодов. Ошибка квантования распространяется на будущие каналы через обратную кривизну. Коды и шкалы пишутся напрямую в Q4_0 payload.

Все статистики сняты с исходного encoder, поэтому это слой-за-слоем оптимизация с фиксированными F16 активациями. Две конфигурации используют вложенные 16 и 64 записи LibriSpeech train-clean-100. Golos проверяет перенос с англоязычной калибровки. Выбор конфигурации ведётся на dev.

[GPTQ](https://arxiv.org/abs/2210.17323) · [whisper.cpp v1.8.3](https://github.com/ggml-org/whisper.cpp/tree/2eeeba56e9edd762b4b38467bab96c2517163158) · [LibriSpeech](https://www.openslr.org/12/) · [Golos Farfield и разбиение](https://huggingface.co/datasets/bond005/sberdevices_golos_100h_farfield#data-splits)

## Воспроизведение

Запускать из этой папки. Нужны Python 3.11, CMake, C++ компилятор и FFmpeg.

```bash
python3.11 -m venv .runtime/venv
.runtime/venv/bin/python -m pip install -r requirements.txt pytest
git clone --depth 1 --branch v1.8.3 https://github.com/ggml-org/whisper.cpp.git .runtime/whisper.cpp
git -C .runtime/whisper.cpp rev-parse HEAD
cmake -S .runtime/whisper.cpp -B .runtime/build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF -DWHISPER_COREML=OFF -DWHISPER_BUILD_SERVER=OFF -DWHISPER_BUILD_TESTS=OFF
cmake --build .runtime/build --target whisper-cli whisper-quantize -j 8

mkdir -p .runtime/models
curl -fL -o .runtime/models/ggml-small.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/80da2d8/ggml-small.bin
shasum -a 256 .runtime/models/ggml-small.bin
.runtime/build/bin/whisper-quantize .runtime/models/ggml-small.bin .runtime/models/small-q4_0.bin q4_0
```

Ожидаемый commit whisper.cpp — `2eeeba56e9edd762b4b38467bab96c2517163158`. SHA-256 F16 — `1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b`. На машине пилота понадобился `-DCMAKE_OSX_SYSROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk`, чтобы использовать совместимую версию macOS SDK.

### Калибровка и экспорт

```bash
.runtime/venv/bin/python fetch_asr.py .runtime/data/libri-cal --split train.100 \
  --offsets 0 3600 7200 10800 14400 18000 21600 25200
.runtime/venv/bin/python fetch_asr.py .runtime/data/libri-dev --split validation --per-offset 4 \
  --offsets 0 320 640 960 1280 1600 1920 2240
.runtime/venv/bin/python fetch_asr.py .runtime/data/golos-dev \
  --dataset bond005/sberdevices_golos_100h_farfield --config default --split validation \
  --per-offset 4 --offsets 0 50 100 150 200 250 300 350

VECLIB_MAXIMUM_THREADS=4 .runtime/venv/bin/python calibrate.py \
  --f16 .runtime/models/ggml-small.bin --control .runtime/models/small-q4_0.bin \
  --manifest .runtime/data/libri-cal/manifest.jsonl --counts 1 \
  --tensor encoder.blocks.0.attn.query.weight --output-dir .runtime/models/single

VECLIB_MAXIMUM_THREADS=4 .runtime/venv/bin/python calibrate.py \
  --f16 .runtime/models/ggml-small.bin --control .runtime/models/small-q4_0.bin \
  --manifest .runtime/data/libri-cal/manifest.jsonl --counts 16 64 --output-dir .runtime/models/full
```

Рядом с каждой моделью записывается JSON со списком тензоров, калибровочными IDs, SHA-256 чекпоинтов, числом кадров и калибровочным MSE. Указанные в манифестах строки данных чередуются между выбранными смещениями. Один и тот же префикс из 16 записей входит в выборку на 64.

Пилотные манифесты фиксируют ревизии источников в поле `asset`. LibriSpeech использует `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`, Golos — `c93949f7140beef4adc404e7b54841e957f81c54`.

### Нативная проверка выбранного слоя

```bash
clang++ -std=c++17 -I.runtime/whisper.cpp/ggml/include tests/ggml_q4_matmul.cpp \
  -L.runtime/build/ggml/src -lggml-base -lggml-cpu \
  -Wl,-rpath,"$PWD/.runtime/build/ggml/src" -o .runtime/ggml_q4_matmul
PYTHONPATH=. .runtime/venv/bin/python tests/verify_ggml_layer.py \
  .runtime/ggml_q4_matmul .runtime/models/single/gptq-q4_0-1.bin \
  encoder.blocks.0.attn.query.weight
.runtime/venv/bin/python -m pytest -q
```

На macOS для компиляции проверки можно указать тот же SDK через `-isysroot`. Проверка сравнивает веса и выход матричного умножения без bias, полученные в Python и GGML CPU.

### Измерения

```bash
.runtime/venv/bin/python benchmark.py --cli .runtime/build/bin/whisper-cli \
  --manifest .runtime/data/libri-dev/manifest.jsonl --language en --threads 4 \
  --output .runtime/results/libri-reproduction \
  --model f16=.runtime/models/ggml-small.bin --model q4_0=.runtime/models/small-q4_0.bin \
  --model gptq16=.runtime/models/full/gptq-q4_0-16.bin \
  --model gptq64=.runtime/models/full/gptq-q4_0-64.bin
```

Для Golos указать манифест `.runtime/data/golos-dev/manifest.jsonl`, язык `ru` и новый каталог измерений. В пилотных замерах Golos порядок был Q4_0, F16, GPTQ-64, GPTQ-16. Декодирование использует greedy, temperature 0, VAD и timestamps выключены. WER и CER агрегируются на уровне корпуса. Для английского используется EnglishTextNormalizer Whisper, для русского BasicTextNormalizer с заменой «ё» на «е». В CER учитываются пробелы. RTF — wall-clock прогонов с загрузкой модели и записью JSON, делённое на суммарную длительность аудио. Peak RSS снимается с процесса whisper-cli. Энергия в пилоте не измерялась.

Для экспериментов с CPU можно указать `--audio-context N` и `--no-flash-attention`. По умолчанию используется полный аудиоконтекст и включённый Flash Attention. Для короткого окна сопоставляйте его покрытие с длительностью записи.

Для итогового сравнения использовать отдельные манифесты LibriSpeech test-clean, test-other и Golos Farfield test с одинаковым набором записей для всех моделей. `benchmark.py` принимает JSONL со строками `{"id":"идентификатор","audio":"путь/к/mono-16k.wav","text":"эталон"}`. Относительные пути к аудио отсчитываются от каталога манифеста. Для калибровки также требуется поле `split` с train-разбиением. Числовой порог WER и параметры итоговых CPU-прогонов зависят от согласованного командой протокола.

## Интеграция с общим экспортом

Точка интеграции на уровне одного тензора — `gptq(weights, second_moment, damp).pack()`. Входная матрица имеет форму `[out_features, in_features]`, а второй момент использует тот же порядок входных каналов. Выход — байты нативного Q4_0 для этого тензора. Общий exporter может направлять такой payload в модель по карте тензоров. Вместе с картой следует сохранить SHA-256 F16 источника, калибровочные IDs и настройки.
