# Русскоязычный бенчмарк: тональность

До этого набора все числа проекта были на английских данных (AG News,
RouterBench), хотя проект русскоязычный. Здесь — воспроизводимый ответ на
вопрос «работает ли конвейер на русском и насколько хорошо».

## 1. Данные

`MonoHime/ru_sentiment_dataset` — агрегат из шести русскоязычных источников
(новости, отзывы, комментарии). Ревизия закреплена:
`a66a66ee3a858c2b59e056b3fb7dfaf501fc5425`.

| Сплит | Строк | neutral | positive | negative |
| --- | ---: | ---: | ---: | ---: |
| train (`--limit 6000`) | 6000 | 1535 | 2884 | 1581 |
| test (validation) | 1500 | 411 | 715 | 374 |

Сплиты автора: `train` и `validation` (последний служит test-срезом).
Перемешивание с `seed=0` обязательно: в исходном файле строки сгруппированы по
источникам, и «первые N» были бы одним доменом.

> **Лицензия.** В карточке датасета она не указана, поэтому веса, обученные на
> нём, **не публикуются** в релизах: при сборке архива пак исключается
> (`EXCLUDE_PACKS=ru-rubert-tiny2`). Публикуются числа, ревизия и команда —
> пак каждый собирает у себя.

## 2. Вопросы

| Вопрос | Тип | Метка |
| --- | --- | --- |
| `sentiment` | choice (`neutral`/`positive`/`negative`) | исходная метка |
| `is_positive` | noul | `positive` → 1 |
| `is_negative` | noul | `negative` → 1 |

Классы несбалансированы, поэтому рядом с accuracy всегда стоит baseline
большинства: «всегда positive» для `sentiment` — 0.477.

## 3. Модель

`cointegrated/rubert-tiny2` (29M параметров, hidden 312) как encoder-only
энкодер, `DECISION_ENCODER_KIND=encoder`, `DECISION_MAX_LENGTH=128`. Головы —
линейные, обучение 600 шагов, `lr=0.05`, val — 20% train, CPU.

## 4. Результаты

| Вопрос | baseline большинства | модель (test) | Δ |
| --- | ---: | ---: | ---: |
| `sentiment` | 0.4767 | **0.5947** | +11.8 п.п. |
| `is_positive` | 0.5233 | **0.7027** | +17.9 п.п. |
| `is_negative` | 0.7507 | **0.8260** | +7.5 п.п. |
| **среднее** | 0.5836 | **0.7078** | +12.4 п.п. |

Train 0.7279, val 0.7011, test 0.7078 — разрыв небольшой, переобучения на 6000
примерах не видно.

## 5. Калибровка

Температура подобрана на val (`training.recipe.calibrate_temperature`), accuracy
не меняется, вероятности становятся честными:

| Вопрос | T | ECE test до | ECE test после | Brier test до → после |
| --- | ---: | ---: | ---: | --- |
| `sentiment` | 2.22 | 0.1758 | **0.0222** | 0.5636 → 0.5105 |
| `is_positive` | 1.87 | 0.0975 | **0.0686** | 0.1959 → 0.1876 |
| `is_negative` | 1.41 | 0.0539 | **0.0170** | 0.1258 → 0.1225 |

Головы были заметно переуверены (ECE `sentiment` 0.18 при accuracy 0.59) —
типичная картина для линейных голов на замороженном энкодере. После
температурного масштабирования `confidence` можно сравнивать с ценой ошибки в
каскаде.

## 6. Воспроизведение

```bash
# данные (нужен интернет: Hugging Face)
.venv/bin/python -m eval.datasets.ru_sentiment \
  --out-dir data/ru_sentiment --limit 6000

# пак (CPU; на GPU те же команды с DECISION_DEVICE=cuda)
DECISION_DEVICE=cpu DECISION_ENCODER_KIND=encoder DECISION_MAX_LENGTH=128 \
  .venv/bin/python -m training.recipe.train_heads --specs ru_sentiment \
  --encoder cointegrated/rubert-tiny2 --dim 312 --limit 6000 \
  --train-steps 600 --lr 0.05 --val-fraction 0.2 --seed 0 \
  --out-dir checkpoints/packs/ru-rubert-tiny2

# калибровка (val/test до и после)
DECISION_DEVICE=cpu DECISION_ENCODER_KIND=encoder DECISION_MAX_LENGTH=128 \
  .venv/bin/python -m training.recipe.calibrate_temperature \
  --specs ru_sentiment --encoder cointegrated/rubert-tiny2 --dim 312 \
  --limit 6000 --checkpoint-dir checkpoints/packs/ru-rubert-tiny2
```

Пак — обычный: `heads.pt` + `metadata.json` + `temperature.json`, подпись
вопросов `047e3380b3abca7f`. Его можно подать серверу как
`DECISION_CHECKPOINT_DIR=checkpoints/packs` вместе с другими доменами: подпись
вопросов не пересекается с AG News и RouterBench.

## 7. Ограничения

- Датасет собран из соцсетей и отзывов; нейтральный класс — 26%, и он самый
  трудный (`sentiment` accuracy 0.59 против 0.83 у `is_negative`).
- Один энкодер (29M) и один seed. Числа — про конвейер на русском, а не про
  предельное качество: `rubert-base`/`ruRoberta` дадут больше, но и весят
  больше.
- Токсичные тексты в датасете встречаются как есть; это влияет на примеры в
  отчётах, но не на метрики.
