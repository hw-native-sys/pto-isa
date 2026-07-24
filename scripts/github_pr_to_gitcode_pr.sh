#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -euo pipefail

GITHUB_EVENT_PATH="${GITHUB_EVENT_PATH:?GITHUB_EVENT_PATH is required}"
GITCODE_TOKEN="${GITCODE_TOKEN:?GITCODE_TOKEN is required}"
GITCODE_OWNER="${GITCODE_OWNER:-cann}"
GITCODE_REPO="${GITCODE_REPO:-pto-isa}"
GITCODE_BASE_BRANCH="${GITCODE_BASE_BRANCH:-master}"
GITCODE_HEAD_OWNER="${GITCODE_HEAD_OWNER:-zhhywang}"
GITCODE_REMOTE_NAME="${GITCODE_REMOTE_NAME:-gitcode}"
GITCODE_PUSH_URL="${GITCODE_PUSH_URL:?GITCODE_PUSH_URL is required}"
GITCODE_API_BASE="${GITCODE_API_BASE:-https://api.gitcode.com/api/v5}"
BRANCH_PREFIX="${BRANCH_PREFIX:-github-pr}"
DRY_RUN="${DRY_RUN:-0}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

die() {
  log "ERROR: $*"
  exit 1
}

json_get() {
  local expr="$1"
  jq -r "${expr}" "${GITHUB_EVENT_PATH}"
}

json_payload() {
  jq -n \
    --arg title "$1" \
    --arg head "$2" \
    --arg base "$3" \
    --arg body "$4" \
    '{"title": $title, "head": $head, "base": $base, "body": $body}'
}

print_pr_url() {
  local response_file="$1"
  local pr_url=""
  local pr_number=""

  pr_url="$(jq -r '.html_url // .web_url // .url // .links.html.href // empty' "${response_file}")"
  if [[ -n "${pr_url}" ]]; then
    log "GitCode PR URL: ${pr_url}"
    return 0
  fi

  pr_number="$(jq -r '.number // .iid // .id // empty' "${response_file}")"
  if [[ -n "${pr_number}" ]]; then
    log "GitCode PR URL: https://gitcode.com/${GITCODE_OWNER}/${GITCODE_REPO}/pulls/${pr_number}"
    return 0
  fi

  log "GitCode PR URL not found in API response"
}

command -v jq >/dev/null 2>&1 || die "jq is required"

pr_number="$(json_get '.pull_request.number')"
pr_title="$(json_get '.pull_request.title')"
pr_body="$(json_get '.pull_request.body // ""')"
pr_html_url="$(json_get '.pull_request.html_url')"
head_sha="$(json_get '.pull_request.head.sha')"
head_repo_full_name="$(json_get '.pull_request.head.repo.full_name')"
head_ref="$(json_get '.pull_request.head.ref')"
base_ref="$(json_get '.pull_request.base.ref')"

[[ -n "${pr_number}" ]] || die "cannot read pull request number"
[[ -n "${head_sha}" ]] || die "cannot read pull request head SHA"

gitcode_branch="${BRANCH_PREFIX}/${pr_number}"
gitcode_head="${GITCODE_HEAD_OWNER}:${gitcode_branch}"
gitcode_title="[GitHub PR #${pr_number}] ${pr_title}"
gitcode_body="$(cat <<EOF
Mirrored from GitHub PR #${pr_number}: ${pr_html_url}

GitHub source branch: ${head_repo_full_name}:${head_ref}
GitHub base branch: ${base_ref}
GitHub head SHA: ${head_sha}

---

${pr_body}
EOF
)"

github_pr_ref="refs/remotes/github-pr/${pr_number}"

log "mirroring GitHub PR #${pr_number} (${head_sha}) to GitCode branch ${gitcode_branch}"

if git remote get-url "${GITCODE_REMOTE_NAME}" >/dev/null 2>&1; then
  git remote set-url "${GITCODE_REMOTE_NAME}" "${GITCODE_PUSH_URL}"
else
  git remote add "${GITCODE_REMOTE_NAME}" "${GITCODE_PUSH_URL}"
fi

git fetch origin "pull/${pr_number}/head:${github_pr_ref}"
fetched_sha="$(git rev-parse "${github_pr_ref}")"
[[ "${fetched_sha}" == "${head_sha}" ]] || die "fetched PR ref ${fetched_sha} does not match event head ${head_sha}"

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1; would push ${github_pr_ref} to ${GITCODE_REMOTE_NAME}/${gitcode_branch}"
  log "DRY_RUN=1; would create GitCode PR '${gitcode_title}' from ${gitcode_head} to ${GITCODE_BASE_BRANCH}"
  exit 0
fi

git push "${GITCODE_REMOTE_NAME}" "${github_pr_ref}:refs/heads/${gitcode_branch}" --force

payload="$(json_payload "${gitcode_title}" "${gitcode_head}" "${GITCODE_BASE_BRANCH}" "${gitcode_body}")"
api_url="${GITCODE_API_BASE}/repos/${GITCODE_OWNER}/${GITCODE_REPO}/pulls"

log "creating GitCode PR from ${gitcode_head} to ${GITCODE_OWNER}/${GITCODE_REPO}:${GITCODE_BASE_BRANCH}"
response_file="$(mktemp)"
trap 'rm -f "${response_file}"' EXIT
status_code="$(
  curl -sS --connect-timeout 30 --max-time 180 \
    -o "${response_file}" -w '%{http_code}' \
    -X POST "${api_url}" \
    -H "Content-Type: application/json" \
    -H "PRIVATE-TOKEN: ${GITCODE_TOKEN}" \
    --data "${payload}"
)"

case "${status_code}" in
  200|201)
    log "GitCode PR created"
    print_pr_url "${response_file}"
    cat "${response_file}"
    ;;
  409|422)
    log "GitCode PR may already exist or request was rejected as duplicate"
    print_pr_url "${response_file}"
    cat "${response_file}"
    ;;
  *)
    cat "${response_file}" >&2
    die "GitCode API returned HTTP ${status_code}"
    ;;
esac
