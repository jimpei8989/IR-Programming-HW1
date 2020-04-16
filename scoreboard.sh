_BASELINE_COLOR="\x1b[38;5;197m"
_STRONG_COLOR="\x1b[38;5;48m"
_MY_COLOR="\x1b[38;5;33m"
_SIMPLE_COLOR="\x1b[38;5;209m"
_NOTHING_COLOR="\x1b[38;5;130m"
_BOLD="\x1b[1m"
_UNDERLINE="\x1b[4m"
_BLINK="\x1b[5m"
_RESET="\x1b[0m"

ME="B06902029"
SLEEP_TIME=30

while true; do
    i=0
    kaggle competitions leaderboard -c wm-2020-vsm-model --show | while IFS= read -r line; do
        if [[ $i -eq 0 ]]; then
            reset
        elif [[ $i -eq 2 ]]; then
            CUR_COLOR=$_STRONG_COLOR
        fi

        data=($line)
        user=${data[1]}

        if [[ ${user} == $ME ]]; then
            echo -e "$_MY_COLOR$_BOLD$_UNDERLINE$line$_RESET"
        elif [[ ${user} == "STRONG_BASELINE.csv" ]]; then
            CUR_COLOR=$_SIMPLE_COLOR
            echo -e "$_BASELINE_COLOR$_BOLD$_BLINK$line$_RESET"
        elif [[ ${user} == "SIMPLE_BASELINE.csv" ]]; then
            CUR_COLOR=$_NOTHING_COLOR
            echo -e "$_BASELINE_COLOR$_BOLD$_BLINK$line$_RESET"
        else
            echo -e "$CUR_COLOR$line$_RESET"
        fi
        i=$((i + 1))
    done
    sleep $SLEEP_TIME
done
