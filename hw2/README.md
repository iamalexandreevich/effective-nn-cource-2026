# ДЗ 2. Воспроизведение DroPE

Воспроизведение статьи **Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings** (Gelberg, Eguchi, Akiba, Cetin; Sakana AI, [arXiv 2512.12167](https://arxiv.org/abs/2512.12167)) на модели SmolLM-360M. Курс «Эффективные NN», ИТМО.

- Отчёт: [report/report.md](report/report.md)
- Полученная модель (веса после рекалибровки): https://huggingface.co/alexander-andreevich/SmolLM-360M-DroPE-66M
- Код авторов статьи: https://github.com/SakanaAI/DroPE (подключён как сабмодуль, не изменялся)

Коротко о результате. Анализ без обучения и базовые методы расширения контекста (PI, NTK, YaRN, обрезка) воспроизведены. Рекалибровка DroPE выполнена на 0.2% бюджета статьи (66 млн токенов, 2×T4 на Kaggle): перплексия падает и температура внимания критична, но качество восстановилось лишь частично, а NIAH на длинном контексте остался нулевым. Подробности, таблицы «в статье / у нас» и список отклонений — в отчёте.

**Известное ограничение.** Числа LongBench в отчёте (приложение) недостоверны: при расчёте на Mac в bf16 внимание `sdpa` давало пустые ответы. Исправление (`eager`) проверено, но полный пересчёт не выполнялся; остальные результаты (перплексия, NIAH, качество) не затронуты. Подробности в отчёте.

## Структура

```
README.md
requirements.txt      окружение для кода и тестов
requirements-eval.txt отдельное окружение для lm-eval-harness (качество в исходном контексте)

src/                  код воспроизведения
  positional.py         режимы RoPE (PI, NTK, YaRN), загрузка DroPE, температура внимания β
  ppl.py                перплексия на заданной длине
  niah.py               стенд needle-in-a-haystack
  long_context_eval.py  оценка метода на 1×, 2×, 4× (перплексия + NIAH)
  longbench.py, longbench_eval.py   LongBench: данные, промпты, метрики и генерация
  tune_beta.py          подбор коэффициента c для β
  quality_eval.py       качество на шести задачах (lm-eval-harness)
  analysis.py, run_analysis.py   позиционное смещение, Fig. 4 и Fig. 6
  pretokenized_data.py, snapshot_callback.py   данные и снимки весов для рекалибровки
  hf_release.py         сборка релиза и карточки модели для Hugging Face
  plot_results.py, report_tables.py            графики и сборка отчёта

configs/              Hydra-конфиг рекалибровки под T4 (подключается к train.py авторов)
kaggle/               запуск рекалибровки на Kaggle
  launch_recalibration.py, run_authors_train.py   лаунчер и обёртка над train.py авторов
  prepare_tokens.py     предварительная токенизация данных
  bundle_code.sh        упаковка кода в датасет Kaggle
  benchmark_throughput.py   замер скорости одной T4
  kernels/{smoke,bench,main,resume}/   ядра Kaggle (метаданные и точка входа)
scripts/              evaluate_snapshots.sh — оценка всех снимков весов; run_longbench.sh — LongBench;
                      check_mps_sdpa.py — проверка ошибки внимания на MPS; publish_to_hf.py — публикация модели
tests/                тесты (pytest)
results/              все числа отчёта в JSON и графики
  eval/ quality/ analysis/ kaggle/
report/               report_template.md → report.md (таблицы собираются из results/)
third_party/DroPE     код авторов (git submodule)
```

Не входит в репозиторий: веса модели (они на Hugging Face и в выводе Kaggle-ядра `lild1tz/drope-main`), кэши данных `artifacts/`, окружения `.venv*`. Команды ниже создают их заново.

## Как запустить

Окружение и тесты:

```bash
git submodule update --init
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src:third_party/DroPE
pytest tests
```

Базовые методы расширения контекста (Mac, MPS; первый запуск скачивает модель и эссе Пола Грэма):

```bash
python src/long_context_eval.py --method yarn --scales 2 4      # также: base, crop, pi, ntk, ntk_dynamic
```

Анализ без обучения:

```bash
python src/run_analysis.py fig6
python src/run_analysis.py fig4
```

Рекалибровка на Kaggle (нужен токен Kaggle в `~/.kaggle/`):

```bash
python kaggle/prepare_tokens.py                                   # токенизация данных
kaggle datasets create -p artifacts/kaggle/tokens_bundle          # первый раз; дальше "datasets version"
bash kaggle/bundle_code.sh
kaggle datasets create -p artifacts/kaggle/code_bundle
kaggle kernels push -p kaggle/kernels/main                        # 2×T4, около 10 часов
kaggle kernels output lild1tz/drope-main -p artifacts/kaggle/main
```

Идентификаторы `lild1tz/...` в `kaggle/kernels/*/kernel-metadata.json` и `kaggle/prepare_tokens.py` относятся к аккаунту автора: для своего запуска замените их.

Оценка снимков, качество и сборка отчёта:

```bash
bash scripts/evaluate_snapshots.sh                                # 2–3 часа на Mac
python -m venv .venv-eval && .venv-eval/bin/pip install -r requirements-eval.txt
PYTHONPATH=src:third_party/DroPE .venv-eval/bin/python src/quality_eval.py \
  --name drope_final --checkpoint artifacts/kaggle/main/snapshots/step-1000 \
  --output results/quality/drope_final.json
python src/plot_results.py && python src/report_tables.py
```

## Загрузка полученной модели

Веса опубликованы на Hugging Face: https://huggingface.co/alexander-andreevich/SmolLM-360M-DroPE-66M (там же карточка модели и ограничения). Загрузчик скачивает их по имени репозитория:

```python
import math, sys
sys.path[:0] = ["src", "third_party/DroPE"]
from positional import load_drope_model, set_attention_temperature

model = load_drope_model("alexander-andreevich/SmolLM-360M-DroPE-66M", beta=1.0, device="mps")
set_attention_temperature(model, 1 + 0.2 * math.log(2))   # для контекста 2×
```

Загрузчик нужен вместо `AutoModel`: он подключает класс авторов и исправляет несовместимость их QKNorm-внимания с `transformers` 4.56.2 (подробности в §7 отчёта).
