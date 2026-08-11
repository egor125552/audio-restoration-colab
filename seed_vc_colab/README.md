# Seed-VC Colab

Эта папка делает две разные вещи: даёт запускаемый Google Colab для Seed-VC и проверяет, что он не сгнил из-за зависимостей.

Upstream закреплён на последнем коммите `Plachtaa/seed-vc` `51383efd921027683c89e5348211d93ff12ac2a8`. Репозиторий Seed-VC архивирован, поэтому плавающая установка из `main` здесь намеренно не используется.

## Что проверяет GitHub Actions

Workflow запускается внутри официального Docker-образа Google Colab `us-docker.pkg.dev/colab-images/public/cpu-runtime:latest` с чистой файловой системой.

Он:

- ставит отдельный Python 3.10;
- клонирует ровно закреплённый Seed-VC;
- устанавливает все upstream-зависимости;
- удаляет только три дублирующие nightly-строки PyTorch из upstream `requirements.txt`, оставляя фиксированные `torch==2.4.0`, `torchvision==0.19.0`, `torchaudio==2.4.0`;
- выполняет `pip check`;
- импортирует настоящие `app.py`, `SeedVCWrapper`, `modules.commons` и основные аудио-библиотеки;
- строит настоящие V1 и V2 Gradio-интерфейсы из upstream-кода;
- запускает этот интерфейс;
- Chromium загружает два настоящих WAV-файла, меняет настройки, нажимает Submit и Clear в V1, затем делает то же в V2;
- Python-бэкенд записывает факт обоих вызовов, поэтому тест не может пройти от одной только красивой HTML-страницы.

В браузерном smoke-тесте дорогой inference callback заменён на безопасный callback, возвращающий загруженный WAV. Это сделано намеренно: GitHub hosted runner не имеет T4. Импорт модели и интерфейс при этом настоящие; только генерация весами не выдаётся за проверенную.

## Что этот тест не доказывает

Он не доказывает, что CUDA-кернелы работают на T4, что V2 помещается в 16 ГБ VRAM и что настоящий diffusion inference завершится. Для этого нужен отдельный запуск на реальной T4.

Официальный Colab Docker `runtime` поддерживает GPU и Google указывает T4 среди протестированных карт, но обычный GitHub hosted runner GPU не предоставляет.

## Colab

Блокнот: `notebooks/Seed_VC_RU.ipynb`.

В Colab выберите T4 GPU, выполните установочную ячейку, затем smoke-import. Последняя ячейка запускает настоящую Seed-VC V2 и скачивает реальные checkpoints.

## Локальный запуск smoke-теста

При наличии Docker:

```bash
docker run --rm --entrypoint /bin/bash \
  -v "$PWD:/workspace" -w /workspace \
  us-docker.pkg.dev/colab-images/public/cpu-runtime:latest \
  -lc 'bash seed_vc_colab/ci_inside_colab.sh'
```
