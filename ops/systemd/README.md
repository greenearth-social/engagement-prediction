# Scheduled BST training on this VM

The service runs as `david.freifeld` in a dedicated clone at
`/home/david.freifeld/greenearth-social/engagement-prediction-training`.
Every invocation fetches `origin/main`, fast-forwards to that revision, runs
`pipenv sync --dev`, then runs `cli.py` in the foreground. It refuses local code
changes, local commits, and branches other than `main`. The revision is printed
in the journal; ClearML also records the pipeline's Git revision.

Keep this clone and its `.venv` exclusively for the service. Do not edit or update
them during a run. Development continues in the existing checkout. Main is the
release boundary for scheduled training: merging code or lockfile changes affects
the next run. Updates to these installed systemd files/helper require reinstalling
them explicitly.

The timer defaults to Sunday at 06:00 UTC. `Persistent=true` triggers one catch-up
run when the timer starts after a missed schedule, using the actual run date.
systemd will not start a second instance while this service is still running.
Failures remain visible in systemd and the journal; there are no automatic retries
or notifications. Models upload through the existing ClearML/GCS pipeline and are
never deployed by this service. Cleanup and model review remain manual.

## Prepare the checkout

First merge the automation changes into `main`. The commands below assume this
VM's existing Python 3.13.11 and Pipenv installation. A separate GPU environment
uses additional disk space; check the root filesystem as well as `/mnt/data`
before creating it. Training output goes to `/mnt/data/dave/outputs/automated-ranker`
and retains the configured data/disk checks.

The service has no interactive shell or SSH agent. As `david.freifeld`, verify
that Git authentication works without one before cloning:

```bash
env -u SSH_AUTH_SOCK GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND='/usr/bin/ssh -o BatchMode=yes' \
  git ls-remote git@github.com:greenearth-social/engagement-prediction.git refs/heads/main
git clone --branch main --single-branch git@github.com:greenearth-social/engagement-prediction.git \
  /home/david.freifeld/greenearth-social/engagement-prediction-training
cd /home/david.freifeld/greenearth-social/engagement-prediction-training
cp config.automation.yml config.automation.local.yml
```

Edit `config.automation.local.yml` for machine-specific overrides, such as output
location, relative date offsets, or training parameters. This ignored file survives
updates; it is the service's only config file and is merged with `cli.py` defaults.
It is initially a copy of `config.automation.yml`, not an overlay on that file.
Leave `posts_end` omitted/null for moving dates. Leave `dry_run` omitted/false for
real runs. Do not set reuse/resume options for a fresh full run.

ClearML and Google credentials must be available to `david.freifeld` without a login
shell. The service sets `HOME` so the existing `~/clearml.conf` and application
default Google credentials can be used. If the pipeline needs other environment
variables, supply them through a systemd service override. Do not put secrets in
the committed unit files.

## Install and test without enabling the schedule

Run from the dedicated checkout. The helper is installed outside the clone so it
cannot replace itself while updating the clone.

```bash
sudo install -D -m 0755 ops/systemd/prepare-checkout.sh /usr/local/libexec/bst-ranker-training-prepare
sudo install -m 0644 ops/systemd/bst-ranker-training.service /etc/systemd/system/
sudo install -m 0644 ops/systemd/bst-ranker-training.timer /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/bst-ranker-training.service /etc/systemd/system/bst-ranker-training.timer
sudo systemctl daemon-reload
systemd-analyze calendar 'Sun *-*-* 06:00:00 UTC'
```

For the first service invocation, use a temporary override to append `--dry-run`.
This still updates the checkout and installs dependencies, but the CLI only prints
the resolved config/dates; it does not create a ClearML task or run any pipeline
stages. It therefore does not validate data access, GPU training, or GCS uploads.

```bash
sudo mkdir -p /run/systemd/system/bst-ranker-training.service.d
sudo tee /run/systemd/system/bst-ranker-training.service.d/dry-run.conf >/dev/null <<'EOF'
[Service]
ExecStart=
ExecStart=/home/david.freifeld/.local/bin/pipenv run python -u cli.py --config config.automation.local.yml --no-background --dry-run
EOF
sudo systemctl daemon-reload
sudo systemctl start bst-ranker-training.service
journalctl -u bst-ranker-training.service -n 200 --no-pager
```

After inspecting the dry-run result, remove just that override and start a real
run manually. `--no-block` returns immediately; the pipeline remains in the
foreground under systemd's supervision.

```bash
sudo rm /run/systemd/system/bst-ranker-training.service.d/dry-run.conf
sudo systemctl daemon-reload
sudo systemctl start --no-block bst-ranker-training.service
journalctl -fu bst-ranker-training.service
```

Confirm that the full run succeeds, the expected dates/checks appear in ClearML,
and model artifacts upload before enabling the schedule:

```bash
sudo systemctl enable --now bst-ranker-training.timer
systemctl list-timers bst-ranker-training.timer
```

## Operate and change the schedule

Use `sudo systemctl start --no-block bst-ranker-training.service` for on-demand
runs so they share systemd's overlap prevention. Independent CLI invocations are
not coordinated with this service.

```bash
systemctl status bst-ranker-training.service bst-ranker-training.timer
systemctl show bst-ranker-training.service -p Result -p ExecMainStatus
journalctl -u bst-ranker-training.service --since yesterday
sudo systemctl disable --now bst-ranker-training.timer
sudo systemctl stop bst-ranker-training.service
```

Disabling the timer prevents future starts; it does not stop a running pipeline.
Stopping the service terminates the current pipeline and retains partial outputs.
After fixing a failure, start the service again or wait for the next scheduled run.

To change the cadence, run `sudo systemctl edit bst-ranker-training.timer` and add,
for example, Monday at 08:00 UTC. The empty `OnCalendar=` clears the original entry:

```ini
[Timer]
OnCalendar=
OnCalendar=Mon *-*-* 08:00:00 UTC
```

Then run `sudo systemctl daemon-reload` and, if the timer is enabled,
`sudo systemctl restart bst-ranker-training.timer`. Inspect the next run with
`systemctl list-timers bst-ranker-training.timer`.
