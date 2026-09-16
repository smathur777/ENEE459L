# ENEE459L

Code repo for course ENEE459L: Embedded AI & IoT Laboratory.

## Updating from upstream

Run these commands from this repository's terminal. Check `git status` first and
commit or stash any unfinished changes before switching branches.

- `upstream` is the TA's repository: `kingston-aditya/ENEE459L`.
- `origin` is your GitHub fork: `smathur777/ENEE459L`.
- locally `main` holds the TA's code; `solutions` holds your answers alongside lab updates.

### 1. Update main with the TA's code

```bash
# Switch to your local main branch.
git switch main

# Download the TA's latest commits and update upstream/main.
# This does not change your local main branch or working files yet.
git fetch upstream

# Advance local main to upstream/main without creating a merge commit.
git merge --ff-only upstream/main

# Upload the updated main branch to your GitHub fork.
git push origin main
```

`--ff-only` stops if your local branch and the other branch have diverged
(each has commits the other lacks).

### 2. Sync your solutions, then bring in the updated main

```bash
# Switch to your local solutions branch.
git switch solutions

# Fetch your fork's solutions branch and advance local solutions to it.
# This brings in work you pushed from another computer, such as the Jetson.
git pull --ff-only origin solutions

# Merge the updated main into solutions, keeping your committed work.
# Git may ask you to resolve conflicts where both branches changed the same code.
git merge main

# After the merge succeeds, upload solutions to your GitHub fork.
git push origin solutions
```