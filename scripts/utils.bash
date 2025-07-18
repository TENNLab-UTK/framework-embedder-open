#!/usr/bin/env bash

# random_float lo hi
# 1 decimal digit
random_float() {
    local lo="${1}"
    local hi="${2}"

    echo $((lo + RANDOM % (hi - lo))).$((RANDOM % 9))
}

random_int() {
    local lo="${1}"
    local hi="${2}"

    echo $((lo + RANDOM % (hi - lo)))
}

random_bool() {
    if ((RANDOM % 2 == 1)); then
        echo true
    else
        echo false
    fi
}

check_equality() {
    n1="${1}"
    n2="${2}"
    precision="${3}"

    if ! awk -v l="${n1}" -v r="${n2}" -v p="${precision}" 'BEGIN { if (sqrt((l-r)**2) <= p) { exit 0 } else { exit 1 }  }'; then
        return 1
    fi

    return 0
}
