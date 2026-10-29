# Деплой, обновление и откат

Сервис ставится в `/opt/decision-model` (переопределяется `DECISION_INSTALL_DIR`),
юнит `decision-api@.service` рендерится под этот каталог и пользователя сервиса.
Ubuntu и CentOS/RHEL-семейство идут одним путём: различия только в установке
пакетов (`apt` против `dnf`), всё остальное — общий `deploy/lib/common.sh`.

## 1. Установка

```bash
# Ubuntu 22.04+ / Debian
sudo bash deploy/ubuntu/install.sh

# CentOS Stream 9 / Rocky 9 / AlmaLinux 9
sudo bash deploy/centos/install.sh
```

Что делает установщик:

1. ставит python3/venv/pip/curl/git;
2. разворачивает **закоммиченное** состояние в `/opt/decision-model`
   (`git clone` из чекаута, из которого запущен скрипт; грязное дерево —
   ошибка, чтобы на сервер не уехало непроверенное);
3. собирает `.venv` из `requirements.txt` уже в каталоге установки;
4. создаёт системного пользователя `decision` без логина и отдаёт ему чтение
   дерева (владелец — `root`, чтобы `upgrade.sh` мог делать `git pull`);
5. рендерит юнит `decision-api@.service` под фактический каталог и
   пользователя и запускает инстанс `api` на `127.0.0.1:8080`.

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `DECISION_INSTALL_DIR` | `/opt/decision-model` | каталог установки |
| `DECISION_SERVICE_USER` | `decision` | пользователь сервиса (создаётся, если нет) |

Если ставить прямо из рабочего дерева (`DECISION_INSTALL_DIR` равен каталогу
чекаута), сервис запускается под пользователем, вызвавшим `sudo`, а права на
исходники не меняются — удобно для одноразовой VM, но не для продакшена.

```bash
# проверить установку, не трогая /opt
sudo DECISION_INSTALL_DIR=/srv/decision-model bash deploy/ubuntu/install.sh
```

Состояние (`.env`, паки, риск-модель) установщик не создаёт: на новом сервере
его привозят из бэкапа (`deploy/restore.sh`) или из релизных паков.

Проверка после установки:

```bash
curl -fsS http://127.0.0.1:8080/health   # liveness: процесс жив
curl -fsS http://127.0.0.1:8080/ready    # readiness: энкодер и головы готовы
curl -fsS http://127.0.0.1:8080/metrics  # Prometheus (см. observability.md)
journalctl -u decision-api@api -n 50 --no-pager
```

`/ready` отвечает `503`, пока сервис не готов; в теле — `reasons`. Для
диагностики причин см. [`troubleshooting.md`](troubleshooting.md), метрики и
логи — [`observability.md`](observability.md).

## 2. Что считается состоянием сервиса

В бэкап входит только то, что нельзя восстановить из релиза или рецепта:

| Путь | Зачем |
| --- | --- |
| `.env` | `DECISION_*`: бэкенд, устройство, порог каскада, пути |
| `checkpoints/packs/` | обученные паки голов (`heads.pt`, `metadata.json`, `temperature.json`) |
| `checkpoints/risk.json` | обученная риск-модель (если включён risk-aware каскад) |

Не входят: `.venv` (пересобирается), веса энкодеров (качаются из HF или
релиза), `checkpoints/finetune/` (воспроизводится рецептом и весит гигабайты).

Архивы складываются в `/var/backups/decision-cascade` (переопределяется
`DECISION_BACKUP_ROOT`). Имя — `decision-cascade-<UTC timestamp>.tar.gz`,
внутри — `.installed-version` с версией на момент бэкапа.

```bash
sudo bash deploy/backup.sh /opt/decision-model
# backup: /var/backups/decision-cascade/decision-cascade-20261029T091500Z.tar.gz
```

## 3. Обновление

```bash
cd /opt/decision-model
sudo bash deploy/upgrade.sh /opt/decision-model
```

Порядок шагов (всё в одном скрипте, одинаково на Ubuntu и CentOS):

1. проверяет, что рабочее дерево git чистое;
2. делает бэкап `.env` + паков + риск-модели;
3. `git fetch --tags --prune` и `git pull --ff-only`;
4. обновляет `.venv` из `requirements.txt`;
5. перезапускает `decision-api@api.service`;
6. ждёт `200` от `/ready` (по умолчанию 60 с).

Если `/ready` не поднялся — скрипт **сам откатывает** данные из бэкапа, код на
предыдущий коммит (`git checkout --detach <old>`), пересобирает venv,
перезапускает сервис и завершается с кодом `1`. После автоматического отката
репозиторий остаётся в detached HEAD на старом коммите — вернуться на ветку:

```bash
sudo git -C /opt/decision-model checkout main
```

Параметры окружения скрипта:

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `DECISION_INSTANCE` | `api` | инстанс юнита `decision-api@<instance>` |
| `DECISION_READY_URL` | `http://127.0.0.1:8080/ready` | куда стучится проверка готовности |
| `DECISION_READY_TIMEOUT` | `60` | сколько секунд ждать `/ready` |
| `DECISION_BACKUP_ROOT` | `/var/backups/decision-cascade` | каталог бэкапов |

## 4. Ручной откат

Если обновление прошло, но качество/поведение не устроило:

```bash
# 1. данные (.env, паки, риск-модель) — из бэкапа
sudo bash deploy/restore.sh /var/backups/decision-cascade/decision-cascade-<stamp>.tar.gz /opt/decision-model

# 2. код — на предыдущий тег
sudo git -C /opt/decision-model checkout v0.13.0

# 3. venv и перезапуск
sudo bash -c 'source deploy/lib/common.sh; ensure_venv /opt/decision-model; restart_service api'

# 4. проверка
curl -fsS http://127.0.0.1:8080/ready
```

`restore.sh` сам перезапускает сервис и ждёт `/ready`; если сервис не поднялся —
печатает команду для `systemctl status` и выходит с ошибкой.

## 5. Паритет Ubuntu/CentOS

| Шаг | Ubuntu | CentOS/RHEL |
| --- | --- | --- |
| Пакеты | `apt-get install python3 python3-venv python3-pip curl git` | `dnf install python3 python3-pip curl git` |
| venv | `python3 -m venv .venv` | то же |
| Каталог установки | `/opt/decision-model` (`DECISION_INSTALL_DIR`) | то же |
| Пользователь сервиса | системный `decision` (`DECISION_SERVICE_USER`) | то же |
| systemd | `systemctl enable --now decision-api@api` | то же |
| upgrade/rollback | `deploy/upgrade.sh`, `deploy/restore.sh` | то же |

Что проверяется автоматически (`tests/test_deploy.py`):

- `distro_from_release` распознаёт `ubuntu`/`debian` и
  `centos`/`rocky`/`almalinux`/`rhel`, а неизвестный дистрибутив — отклоняет;
- бэкап и восстановление `.env`, паков и риск-модели на временном каталоге;
- `shellcheck -x` для всех `deploy/**/*.sh`;
- `render_unit` подставляет в юнит фактический каталог и пользователя и
  отклоняет нераскрытые placeholder'ы;
- `ensure_project_dir` разворачивает только закоммиченное состояние, не
  копирует `.venv`/`checkpoints` и не перезаписывает непустой каталог;
- systemd-юнит: `KillSignal=SIGTERM`, `TimeoutStopSec=30`,
  `--timeout-graceful-shutdown 30`.

Реальный прогон на VM не автоматизирован: скрипты требуют root и systemd.
Перед выкаткой на новый дистрибутив прогоните `install.sh` → `upgrade.sh` →
`restore.sh` на одноразовой VM и сверьте `/ready`.

## 6. Аккуратное завершение

`systemctl restart`/`stop` посылает `SIGTERM`. uvicorn перестаёт принимать новые
соединения, но доводит уже принятые запросы; `TimeoutStopSec=30` — верхняя
граница. Контракт закреплён тестом `tests/test_graceful_shutdown.py`: запрос,
начатый до `SIGTERM`, получает `200`, а не обрыв.
