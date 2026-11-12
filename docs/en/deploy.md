# Deploy, upgrade and rollback

**English** · [Русский](../deploy.md)

The service is installed into `/opt/decision-model` (overridable with `DECISION_INSTALL_DIR`), and the `decision-api@.service` unit is rendered for that directory and the service user. Ubuntu and the CentOS/RHEL family follow one path: only package installation differs (`apt` vs `dnf`), everything else is the shared `deploy/lib/common.sh`.

## 1. Installation

```bash
# Ubuntu 22.04+ / Debian
sudo bash deploy/ubuntu/install.sh

# CentOS Stream 9 / Rocky 9 / AlmaLinux 9
sudo bash deploy/centos/install.sh
```

What the installer does:

1. installs Python 3.11+ (Ubuntu — `python3`/`venv`, CentOS — `python3.11`), pip, curl and git;
2. deploys the **committed** state into `/opt/decision-model` (`git clone` from the checkout the script runs in; a dirty tree is an error, so unverified code never reaches the server);
3. builds `.venv` from `requirements.txt` already inside the installation directory;
4. creates a `decision` system user without a login and gives it read access to the tree (owner is `root`, so `upgrade.sh` can `git pull`);
5. renders the `decision-api@.service` unit for the actual directory and user and starts the `api` instance on `127.0.0.1:8080`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DECISION_INSTALL_DIR` | `/opt/decision-model` | installation directory |
| `DECISION_SERVICE_USER` | `decision` | service user (created if missing) |

If you install straight from a working tree (`DECISION_INSTALL_DIR` equals the checkout directory), the service runs as the user who invoked `sudo` and source permissions are not changed — handy for a one-off VM, not for production.

```bash
# check the installation without touching /opt
sudo DECISION_INSTALL_DIR=/srv/decision-model bash deploy/ubuntu/install.sh
```

The installer does not create state (`.env`, packs, risk model): on a new server it comes from a backup (`deploy/restore.sh`) or from the release packs. Without `.env` the service starts on defaults — the file is optional, so "forgot to copy it" does not turn into a broken service.

Post-install checks:

```bash
curl -fsS http://127.0.0.1:8080/health   # liveness: the process is alive
curl -fsS http://127.0.0.1:8080/ready    # readiness: encoder and heads are ready
curl -fsS http://127.0.0.1:8080/metrics  # Prometheus (see observability.md)
journalctl -u decision-api@api -n 50 --no-pager
```

`/ready` answers `503` while the service is not ready; the body contains `reasons`. For diagnosing the causes see [`troubleshooting.md`](troubleshooting.md); metrics and logs are in [`observability.md`](../observability.md).

## 2. What counts as service state

A backup includes only what cannot be restored from a release or a recipe:

| Path | Why |
| --- | --- |
| `.env` | `DECISION_*`: backend, device, cascade threshold, paths. Read by the unit (`EnvironmentFile=-<project>/.env`), sample in `.env.example` |
| `checkpoints/packs/` | trained head packs (`heads.pt`, `metadata.json`, `temperature.json`) |
| `checkpoints/risk.json` | trained risk model (if the risk-aware cascade is enabled) |

Not included: `.venv` (rebuilt), encoder weights (downloaded from HF or a release), `checkpoints/finetune/` (reproduced by the recipe and weighs gigabytes).

Archives are stored in `/var/backups/decision-cascade` (overridable with `DECISION_BACKUP_ROOT`). The name is `decision-cascade-<UTC timestamp>.tar.gz`, and inside there is `.installed-version` with the version at backup time.

```bash
sudo bash deploy/backup.sh /opt/decision-model
# backup: /var/backups/decision-cascade/decision-cascade-20261029T091500Z.tar.gz
```

## 3. Upgrade

```bash
cd /opt/decision-model
sudo bash deploy/upgrade.sh /opt/decision-model
```

The order of steps (all in one script, identical on Ubuntu and CentOS):

1. checks that the git working tree is clean;
2. backs up `.env` + packs + risk model;
3. `git fetch --tags --prune` and `git pull --ff-only`;
4. updates `.venv` from `requirements.txt`;
5. restarts `decision-api@api.service`;
6. waits for `200` from `/ready` (60 s by default).

If `/ready` does not come up, the script **rolls back by itself**: data from the backup, code to the previous commit (`git checkout --detach <old>`), rebuilds the venv, restarts the service and exits with code `1`. After an automatic rollback the repository stays in detached HEAD on the old commit — to return to the branch:

```bash
sudo git -C /opt/decision-model checkout main
```

Script environment parameters:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DECISION_INSTANCE` | `api` | instance of the `decision-api@<instance>` unit |
| `DECISION_READY_URL` | `http://127.0.0.1:8080/ready` | where the readiness check goes |
| `DECISION_READY_TIMEOUT` | `60` | how many seconds to wait for `/ready` |
| `DECISION_BACKUP_ROOT` | `/var/backups/decision-cascade` | backup directory |

## 4. Manual rollback

If the upgrade succeeded but quality/behaviour is not acceptable:

```bash
# 1. data (.env, packs, risk model) — from the backup
sudo bash deploy/restore.sh /var/backups/decision-cascade/decision-cascade-<stamp>.tar.gz /opt/decision-model

# 2. code — to the previous tag
sudo git -C /opt/decision-model checkout v0.13.0

# 3. venv and restart
sudo bash -c 'source deploy/lib/common.sh; ensure_venv /opt/decision-model; restart_service api'

# 4. check
curl -fsS http://127.0.0.1:8080/ready
```

`restore.sh` restarts the service itself and waits for `/ready`; if the service does not come up, it prints the command for `systemctl status` and exits with an error.

## 5. Ubuntu/CentOS parity

| Step | Ubuntu | CentOS/RHEL |
| --- | --- | --- |
| Packages | `apt-get install python3 python3-venv python3-pip curl git` | `dnf install python3.11 python3.11-pip curl git` |
| venv | `python3 -m venv .venv` | `python3.11 -m venv .venv` |
| Install directory | `/opt/decision-model` (`DECISION_INSTALL_DIR`) | the same |
| Service user | system `decision` (`DECISION_SERVICE_USER`) | the same |
| Python | 3.11+ (Ubuntu 24.04+; on 22.04 — `python3.11`) | 3.11+ (`python3.11` from AppStream) |
| systemd | `systemctl enable --now decision-api@api` | the same |
| upgrade/rollback | `deploy/upgrade.sh`, `deploy/restore.sh` | the same |

What is checked automatically (`tests/test_deploy.py`):

- `distro_from_release` recognises `ubuntu`/`debian` and `centos`/`rocky`/`almalinux`/`rhel`, and rejects an unknown distribution;
- backup and restore of `.env`, packs and the risk model in a temporary directory;
- `shellcheck -x` for every `deploy/**/*.sh`;
- `render_unit` substitutes the actual directory and user into the unit and rejects unexpanded placeholders;
- `ensure_project_dir` deploys only the committed state, does not copy `.venv`/`checkpoints` and does not overwrite a non-empty directory;
- systemd unit: `KillSignal=SIGTERM`, `TimeoutStopSec=30`, `--timeout-graceful-shutdown 30`.

A live VM run is not automated: the scripts require root and systemd. Before rolling out to a new distribution, run `install.sh` → `upgrade.sh` → `restore.sh` on a disposable VM and check `/ready`.

The interpreter is chosen once and identically for install and upgrade: `DECISION_PYTHON`, otherwise the first available `python3.13`/`python3.12`/`python3.11`/`python3` with version ≥ 3.11. If none fits, the script fails with a hint (`dnf install python3.11`) instead of building a venv on 3.9. On Ubuntu 22.04 `python3` is 3.10, so you need `python3.11` (deadsnakes) or 24.04.

Verified live on both distributions: `install.sh` → `/health`, `/ready`, `/metrics`, `X-Request-ID`, `POST /decisions` → `upgrade.sh` → `backup.sh` → corrupt `.env` → `restore.sh` → teardown — **PASS**. The unit is rendered for `/opt/decision-model`, the service runs as the `decision` user, and after teardown neither the unit, nor the directory, nor the user remain.

| Date | Distribution | Version | What was checked |
| --- | --- | --- | --- |
| 2026-10-29 | Ubuntu 24.04.5, systemd 255 | `0.15.0` | full cycle |
| 2026-11-10 | CentOS Stream 9 (cloud VM) | `1.0.0` (`11fe196`) | full cycle, `python3.11` from AppStream |

The CentOS run closed a real defect: `python3` there is 3.9, and the installation built the venv on it. Now `python_bin` requires ≥ 3.11 and hints at `dnf install python3.11`; on Stream 9 it installs `python3.11` (3.11.13).

## 6. Graceful shutdown

`systemctl restart`/`stop` sends `SIGTERM`. uvicorn stops accepting new connections but finishes requests already accepted; `TimeoutStopSec=30` is the upper bound. The contract is pinned by `tests/test_graceful_shutdown.py`: a request started before `SIGTERM` gets `200`, not a connection reset.
