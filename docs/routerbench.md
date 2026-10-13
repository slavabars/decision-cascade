# Публичный бенчмарк RouterBench (маршрутизация LLM)

Второй публичный адаптер после AG News — **RouterBench** (Hu et al., 2024,
[arXiv:2403.12031](https://arxiv.org/abs/2403.12031),
[`withmartian/routerbench`](https://huggingface.co/datasets/withmartian/routerbench)).
Это ~36k промптов из стандартных наборов (MMLU, GSM-8K, HellaSwag, MBPP,
WinoGrande, MT-Bench, ...), для каждого — оценка качества (0..1) и стоимость
ответа 11 разных LLM. Это не классификация тем, а **настоящая decision-задача**:
«стоит ли эскалировать запрос на более сильную (и дорогую) модель». Поэтому
набор подходит для честной доменной калибровки каскада.

Датасет — pickle-таблицы (`routerbench_0shot.pkl`, `routerbench_5shot.pkl`);
версия зафиксирована ревизией `ROUTERBENCH_REVISION`, чтобы числа были
воспроизводимы.

## Маппинг на типизированные вопросы

Берём три уровня моделей по стоимости:

| Уровень | Модель |
| --- | --- |
| `cheap` | `mistralai/mistral-7b-chat` |
| `mid` | `mistralai/mixtral-8x7b-chat` |
| `strong` | `gpt-4-1106-preview` |

Из оценок этих трёх моделей детерминированно выводятся все три типа решений:

| Вопрос | Тип | Метка |
| --- | --- | --- |
| `escalate` | `noul` | 1, если дешёвая модель не решила задачу полностью (`cheap < 1`) — триггер каскада |
| `route` | `choice` | уровень с максимальным качеством; при равенстве — самый дешёвый (`cheap`/`mid`/`strong`) |
| `benefit` | `score` | выигрыш сильной модели над дешёвой (`strong - cheap`): `none`/`some`/`large` |

Правила описаны рядом с `_escalate_label`, `_route_label`, `_benefit_label` в
`eval/datasets/routerbench.py`. Промпт (`str`-представление списка сообщений)
разворачивается в текст состояния функцией `parse_prompt`.

Готового train/test в датасете нет, поэтому разбиение детерминированное (по
`seed`) и берётся из перемешанного порядка (случайная подвыборка, а не срез по
`eval_name`). По умолчанию `test_fraction = 0.25`.

## Материализация

```bash
.venv/bin/python -m eval.datasets.routerbench --out-dir data/routerbench --limit 6000
# train: 6000 -> data/routerbench/train.jsonl
# test : 1500 -> data/routerbench/test.jsonl
```

## Честный отчёт (untrained vs trained)

```bash
DECISION_MAX_LENGTH=256 .venv/bin/python -m eval.report.run \
  --specs routerbench --encoder openai-community/gpt2 --dim 768 \
  --train-steps 600 --device cuda --limit 4000 \
  --out-dir data/report_routerbench
```

Референс (gpt2, GPU, `--limit 4000 --train-steps 600`):

| `question` | `type` | acc (off) | acc (on) | ECE (off) | ECE (on) |
| --- | --- | --- | --- | --- | --- |
| `escalate` | noul | 0.214 | 0.787 | 0.786 | 0.213 |
| `route` | choice | 0.398 | 0.424 | 0.598 | 0.434 |
| `benefit` | score | 0.456 | 0.568 | 0.349 | 0.408 |
| **OVERALL** | | **0.356** | **0.593** | | |

`metrics.json` и `reliability_*.png` кладутся в `--out-dir`.

## Честная оговорка

Маршрутизация — **трудная** задача, и результат это подтверждает:

- `escalate` выходит примерно на **базовую частоту класса** (~0.79): голова
  ловит распространённость эскалаций, а не сам сигнал. Brier ≈ 0.213 против
  `p(1-p) ≈ 0.166` у константного предиктора базовой частоты — то есть навыка
  сверх «всегда эскалировать» почти нет.
- `route` (3 класса, chance ≈ 0.333) и `benefit` дают **небольшой** прирост над
  majority-классом.
- Увеличение выборки и шагов (`--limit 16000 --train-steps 1500`) качество
  `escalate`/`route` **не улучшает** — узкое место в заморожённых признаках, а не
  в объёме данных.

Вывод: RouterBench ценен как честный публичный бенчмарк (`accuracy < 1.0`, а не
`1.000` на синтетике) и как проверка калибровки, но заморожённый gpt2 с линейными
головами — не потолок качества. Улучшение (fine-tuning энкодера, более крупная
модель, признаки о сложности запроса) — отдельная задача.

Примечание: если раннер не на GPU, уберите `--device cuda`; на CPU прогон
заметно дольше (энкодер — узкое место).
