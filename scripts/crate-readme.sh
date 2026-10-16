#!/bin/sh
# Writes lib/README.md, the README crates.io shows for media-pp, from the
# repository's own README.md.
#
# The two differ only in their links. README.md links the repository's
# files relatively, so that GitHub opens them on the branch being read —
# main's guide beside main's README. crates.io resolves a relative link
# against the crate's directory, lib/, where none of them are, and against
# the default branch, which can be ahead of the release: a guide using API
# not yet published. So lib/README.md links each of those files at the tag
# of the version lib/Cargo.toml says, and a reader who came from crates.io
# reads the documentation of the version they installed.
#
#   scripts/crate-readme.sh          write lib/README.md
#   scripts/crate-readme.sh --check  fail if lib/README.md is not what this
#                                    would write (CI runs this)
#
# Edit README.md, never lib/README.md, and run this after changing either
# README.md or the version. Reference-style links only: an inline link to a
# repository file is refused, since rewriting one inside running text is
# where a script like this goes wrong.

set -eu

check=0
case ${1:-} in
    --check) check=1 ;;
    "") ;;
    *)
        echo "usage: $0 [--check]" >&2
        exit 2
        ;;
esac

root=$(cd "$(dirname "$0")/.." && pwd)
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' "$root/lib/Cargo.toml" | head -n 1)
repository=$(sed -n 's/^repository = "\(.*\)"$/\1/p' "$root/lib/Cargo.toml" | head -n 1)
if [ -z "$version" ] || [ -z "$repository" ]; then
    echo "no version or repository in lib/Cargo.toml" >&2
    exit 1
fi

written=$(mktemp)
trap 'rm -f "$written"' EXIT

awk -v root="$root" -v base="$repository" -v tag="v$version" '
    BEGIN {
        print "<!-- Written by scripts/crate-readme.sh from README.md, with its"
        print "     links at the " tag " tag. Edit README.md, not this file. -->"
        print ""
    }
    # A link to a page elsewhere, or within this page, is left as it is.
    function relative(target) {
        return target !~ /^(https?:|mailto:|#)/
    }
    {
        line = $0
        rest = line
        while (match(rest, /\]\([^)]+\)/)) {
            target = substr(rest, RSTART + 2, RLENGTH - 3)
            if (relative(target)) {
                printf "README.md:%d: inline link to %s; make it a reference-style link\n", NR, target > "/dev/stderr"
                failed = 1
            }
            rest = substr(rest, RSTART + RLENGTH)
        }
        if (match(line, /^\[[^:]+\]: /)) {
            label = substr(line, 1, RLENGTH)
            target = substr(line, RLENGTH + 1)
            if (relative(target)) {
                path = target
                sub(/#.*/, "", path)
                kind = system("test -d \"" root "/" path "\"") == 0 ? "tree" : "blob"
                if (kind == "blob" && system("test -f \"" root "/" path "\"") != 0) {
                    printf "README.md:%d: %s is not in the repository\n", NR, path > "/dev/stderr"
                    failed = 1
                }
                line = label base "/" kind "/" tag "/" target
            }
        }
        print line
    }
    END { exit failed }
' "$root/README.md" > "$written"

if [ $check -eq 1 ]; then
    if ! cmp -s "$written" "$root/lib/README.md"; then
        echo "lib/README.md is not what README.md and version $version make:" >&2
        diff "$root/lib/README.md" "$written" >&2 || true
        echo "run scripts/crate-readme.sh and commit lib/README.md" >&2
        exit 1
    fi
    exit 0
fi

cp "$written" "$root/lib/README.md"
echo "wrote lib/README.md, its links at v$version"
