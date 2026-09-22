# launcher_selfcopy.sh — source this at the TOP of any launcher. Bash reads a script
# incrementally by byte offset while it runs, so editing the repo copy of a live
# launcher can make it execute garbage from that offset onward.
#
# . notebooks/scripts/launcher_selfcopy.sh || { echo "missing"; exit 1; }
#
# It copies the launcher to <dir>/launched_<name>.sh and execs that copy: the repo copy
# can then be edited freely and the run continues from its own snapshot.
#
# Source it before any `shift` or `set --`. The re-exec passes "$@", so a positional
# consumed above this line is gone from the run that actually happens.
#
# Reading "$1" is safe and does not need to wait; only `shift` and `set --` are.
#
# The snapshot goes to $SELFCOPY_DIR, else $OUTDIR if the launcher set one before
# sourcing, else <repo>/outputs/logs/_launched.
if [ -z "${LAUNCHER_SELFCOPY_DONE:-}" ]; then
    _self=${BASH_SOURCE[1]:-$0}
    _name=$(basename "$_self")
    _out=${SELFCOPY_DIR:-${OUTDIR:-}}
    if [ -z "$_out" ]; then
        _out="$(cd "$(dirname "$_self")/../.." && pwd)/outputs/logs/_launched"
    fi
    mkdir -p "$_out"
    _copy="$_out/launched_$_name"
    if [ "$(cd "$(dirname "$_self")" && pwd)/$_name" != "$(cd "$(dirname "$_copy")" && pwd)/$(basename "$_copy")" ]; then
        cp "$_self" "$_copy"
        _md5_repo=$( (md5 -q "$_self" 2>/dev/null || md5sum "$_self" | cut -d' ' -f1) )
        _md5_copy=$( (md5 -q "$_copy" 2>/dev/null || md5sum "$_copy" | cut -d' ' -f1) )
        cat > "$_out/launched_${_name%.sh}.md5.json" <<JSON
{
  "launcher": "$_name",
  "repo_path": "$_self",
  "executed_copy": "$_copy",
  "md5_repo_at_launch": "$_md5_repo",
  "md5_executed_copy": "$_md5_copy",
  "identical_at_launch": $( [ "$_md5_repo" = "$_md5_copy" ] && echo true || echo false ),
  "argc_at_selfcopy": $#,
  "argv_at_selfcopy": "$*",
  "why": "the executed copy is immune to edits of the repo copy while the run is live; argv is recorded because the re-exec passes \"\$@\" and a positional consumed above the source line would be lost"
}
JSON
        if [ -n "${LAUNCHER_SELFCOPY_PROBE:-}" ]; then
            # Test mode: the snapshot and its record are written, then report what
            # would be re-exec'd and stop. Nothing below the source line runs.
            echo "[launcher-probe] $_name argc=$# argv=[$*]"
            exit 0
        fi
        echo "[launcher] executing $_copy (md5 $_md5_copy); repo copy may be edited freely"
        export LAUNCHER_SELFCOPY_DONE=1
        export LAUNCHER_SELFCOPY_ARGV="$*"
        export LAUNCHER_SELFCOPY_ARGC=$#
        exec bash "$_copy" "$@"
    fi
else
    # This IS the re-exec'd copy. Assert the arguments survived the exec, so a future
    # edit to the exec line fails here instead of silently changing what the run does.
    if [ -n "${LAUNCHER_SELFCOPY_ARGC:-}" ] \
       && { [ "$#" != "$LAUNCHER_SELFCOPY_ARGC" ] || [ "$*" != "${LAUNCHER_SELFCOPY_ARGV:-}" ]; }; then
        echo "launcher_selfcopy: ARGUMENTS CHANGED ACROSS THE RE-EXEC — refusing to run." >&2
        echo "  before: argc=$LAUNCHER_SELFCOPY_ARGC argv=[${LAUNCHER_SELFCOPY_ARGV:-}]" >&2
        echo "  after : argc=$# argv=[$*]" >&2
        exit 1
    fi
    # These three are exported, so a launcher that runs another launcher would hand
    # the child the parent's argv and the child's own check would refuse. Clear them.
    unset LAUNCHER_SELFCOPY_DONE LAUNCHER_SELFCOPY_ARGV LAUNCHER_SELFCOPY_ARGC
fi
