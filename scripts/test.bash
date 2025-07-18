#!/usr/bin/env bash

if [ $# -ne 2 ]; then
    echo 'usage: bash scripts/test.bash [processor] yes|no(keep temporary files)' >&2
    exit 1
fi

processor="${1}"
if [ "${processor}" != risp ] && [ "${processor}" != rispSoA ]; then
    echo "Procesor ${processor} not supported"
    echo "    Supported Processors: (risp rispSoA)"
fi

keep="${2}"
if [ "${keep}" != yes -a "${keep}" != no ]; then
    echo 'keep parameter must be "yes" or "no"' >&2
    exit 1
fi

# Checking for full framework
if [ -z "${fr}" ]; then
    printf 'Cannot find your fr (framework) environment variable.\n' 2>&1
    exit 1
fi

# Make testing scratch directory
if ! [ -d testing_scratch ]; then
    mkdir testing_scratch
fi

test_harness=$(
    cat <<'EOF'
#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <sstream>

using namespace std;

int main(int argc, char* argv[]) {
    vector<vector<int>> output_raster;
    string line;
    while (getline(cin, line)) {
        stringstream ss(line);
        string tmp;
        vector<string> tokens;

        while (ss >> tmp) {
            tokens.push_back(tmp);
        }

        if (tokens.size() == 0) {
            continue;
        } else if (tokens[0] == "ML" || tokens[0] == "GT" || tokens[0] == "NCH") {
            continue;
        } else if (tokens[0] == "Q") {
            return 0;
        } else if (tokens[0] == "ASR") {
            int neuron = stoi(tokens[1]);
            int input_ind;
            for (size_t i = 0; i < NUM_INPUT_NEURONS; i++) {
                if (INPUT_IND_TO_NEURON_IND[i] == neuron) {
                    input_ind = i;
                    break;
                }
            }

            for (size_t i = 0; i < tokens[2].size(); i++) {
                apply_spike(input_ind, i, tokens[2][i] == '1');
            }
        } else if (tokens[0] == "AS") {
            for (size_t i = 1; i < tokens.size(); i+=3) {
                int input_ind;
                for (size_t j = 0; j < NUM_INPUT_NEURONS; j++) {
                    if (INPUT_IND_TO_NEURON_IND[j] == stoi(tokens[i])) {
                        input_ind = j;
                        break;
                    }
                }

                apply_spike(input_ind, stoi(tokens[i+1]), stof(tokens[i+2]));
            }
        } else if (tokens[0] == "RUN") {
            output_raster.clear();
            output_raster.resize(NUM_OUTPUT_NEURONS);
            for (size_t i = 0; i < stoi(tokens[1]); i++) {
                run(STEP);

                for (size_t j = 0; j < NUM_OUTPUT_NEURONS; j++) {
                    output_raster[j].push_back(output_count(j));
                }
            }
        } else if (tokens[0] == "GSR") {
            for (size_t i = 0; i < NUM_OUTPUT_NEURONS; i++) {
                printf("%-6u : ", OUTPUT_IND_TO_NEURON_IND[i]);
                for (size_t j = 0; j < output_raster[i].size(); j++) {
                    printf("%d", output_raster[i][j]);
                }
                printf("\n");
            }
        } else if (tokens[0] == "OC") {
            for (size_t i = 0; i < NUM_OUTPUT_NEURONS; i++) {
                unsigned int output_spikes = 0;

                for (size_t j = 0; j < output_raster[i].size(); j++) {
                    output_spikes += output_raster[i][j];
                }

                printf("%-6u : %u\n", OUTPUT_IND_TO_NEURON_IND[i], output_spikes);
            }
        } else if (tokens[0] == "OT") {
            for (size_t i = 0; i < NUM_OUTPUT_NEURONS; i++) {
                printf("%-6u : ", OUTPUT_IND_TO_NEURON_IND[i]);

                for (size_t j = 0; j < output_raster[i].size(); j++) {
                    if (output_raster[i][j] == 1) {
                        printf("%.1f ", (float)j);
                    }
                }

                printf("\n");
            }
        } else if (tokens[0] == "CLEAR-A" || tokens[0] == "CA") {
            clear_activity();
        } else {
            printf("Unsupported command %s\n", tokens[0].c_str());
        }
    }
}
EOF
)

for test_dir in testing/*; do
    # Store label
    label=$(cat "${test_dir}"/label.txt)

    # Get correct proc params
    bash "${test_dir}"/processor.sh >tmp_proc_params.json

    # Get empty net
    (
        echo M risp tmp_proc_params.json
        echo EMPTYNET tmp_empty_network.txt
    ) | "${fr}"/cpp-apps/bin/processor_tool_risp

    # Get full net
    "${fr}"/bin/network_tool <"${test_dir}"/network_tool.txt >tmp_nt_output.txt 2>&1
    if [ $(wc tmp_nt_output.txt | awk '{ print $1 }') != 0 ]; then
        echo "Test ${test_dir} - ${label}" >&2
        echo "There was an error in the network_tool command when I ran:" >&2
        echo "" >&2
        echo "${fr}/bin/network_tool < ${test_dir}/network_tool.txt > tmp_nt_output.txt" >&2
        echo "" >&2
        cat tmp_nt_output.txt >&2
        exit 1
    fi

    # Now input the appropriate commands
    cp "${test_dir}"/processor_tool.txt tmp_pt_input.txt

    # Generate C program
    bin/framework_embedder -p "${processor}" <tmp_network.txt >testing_scratch/GENERATED.c

    # For runtime inclusive we step 0, otherwise 1
    step=1
    if [ "$(jq '.run_time_inclusive' tmp_proc_params.json)" = 'true' ]; then
        step=0
    fi

    (
        sed -e '/#define NUM_SYN/c #define NUM_SYNAPSES 1600' -e '/#define MAX_NU/c #define MAX_NUM_TIMESTEPS 1600' testing_scratch/GENERATED.c
        echo "#define STEP ${step}"
        echo "${test_harness}"
    ) >testing_scratch/GENERATED_FULL.cpp

    # Compile with driver code
    clang++ testing_scratch/GENERATED_FULL.cpp -o testing_scratch/a.out

    testing_scratch/a.out <"${test_dir}"/processor_tool.txt >tmp_pt_output.txt

    dif=$(diff tmp_pt_output.txt "${test_dir}"/correct_output.txt | wc | awk '{ print $1 }')
    if [ "${dif}" != 0 ]; then
        echo "Test ${test_dir} - ${label}" >&2
        echo "Error: Output does not match the correct output." >&2
        echo "       Ouptut file is tmp_pt_output.txt" >&2
        echo "       Correct output file is ${test_dir}/correct_output.txt" >&2
        exit 1
    fi

    echo "Passed Test ${test_dir} - ${label}"
    if [ "${keep}" = no ]; then
        rm -f tmp_proc_params.txt \
            tmp_network.txt \
            tmp_nt_output.txt \
            tmp_pt_output.txt \
            tmp_pt_input.txt \
            tmp_pt_error.txt \
            tmp_empty_network.txt
    fi
done
