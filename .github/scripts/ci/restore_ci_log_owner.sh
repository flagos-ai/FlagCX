#!/bin/sh
# Return the dedicated log mount to the invoking host runner.
set -eu

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <log-directory> <host-uid> <host-gid>" >&2
  exit 2
fi
log_root=$1
host_uid=$2
host_gid=$3
for owner_id in "$host_uid" "$host_gid"; do
  case "$owner_id" in
    ''|*[!0-9]*) echo "Log owner UID/GID must be numeric" >&2; exit 2 ;;
  esac
done
if [ ! -d "$log_root" ] || [ -L "$log_root" ] ||
    [ "$(cd "$log_root" && pwd -P)" = / ]; then
  echo "Expected a dedicated log directory: $log_root" >&2
  exit 2
fi

# Do not follow links out of the log tree or broaden other users' permissions.
find -P "$log_root" -type d -exec chmod u+rwx -- {} +
find -P "$log_root" -type f -exec chmod u+rwX -- {} +
chown -hRP -- "$host_uid:$host_gid" "$log_root"
