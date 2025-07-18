#include "EmbeddedRispSoA.hpp"
#include "helpers.hpp"
#include <unordered_map>

EmbeddedRispSoA::EmbeddedRispSoANetwork::EmbeddedRispSoANetwork(
    neuro::Network *net, double _spike_value_factor, double _min_potential,
    char leak, bool _run_time_inclusive, bool _threshold_inclusive,
    bool _fire_like_ravens, bool _discrete, bool _inputs_from_weights,
    uint32_t _noisy_seed, double _noisy_stddev, vector<double> &_weights,
    vector<double> &_stds)
    : risp::Network(net, _spike_value_factor, _min_potential, leak,
                    _run_time_inclusive, _threshold_inclusive,
                    _fire_like_ravens, _discrete, _inputs_from_weights,
                    _noisy_seed, _noisy_stddev, _weights, _stds) {
    sorted_neuron_vector_public = sorted_neuron_vector;
}

EmbeddedRispSoA::EmbeddedRispSoA(neuro::json &params)
    : risp::Processor(params) {
    rnet = nullptr;
    enet = nullptr;
}

EmbeddedRispSoA::~EmbeddedRispSoA() {
    if (enet != nullptr) {
        delete enet;
    }
}

bool EmbeddedRispSoA::load_network(neuro::Network *net, int network_id) {
    bool res;

    res = risp::Processor::load_network(net, network_id);

    rnet = net;
    enet = new EmbeddedRispSoANetwork(
        net, spike_value_factor, min_potential, leak_mode[0],
        run_time_inclusive, threshold_inclusive, fire_like_ravens, discrete,
        inputs_from_weights, noisy_seed, noisy_stddev, weights, stds);

    return res;
}

std::string EmbeddedRispSoA::gen_static_c(unsigned int max_num_timesteps) {
    IndentString s;
    risp::Neuron *cur_neuron;
    neuro::Node *cur_node;
    std::unordered_map<uint32_t, unsigned int> neuron_id_to_ind;
    std::vector<unsigned int> input_ind_to_neuron_ind;
    std::vector<unsigned int> output_ind_to_neuron_ind;
    unsigned int i;
    unsigned int j;
    unsigned int max_outgoing;

    max_outgoing = 0;
    for (i = 0; i < enet->sorted_neuron_vector_public.size(); i++) {
        cur_neuron = enet->sorted_neuron_vector_public[i];

        if (cur_neuron->synapses.size() > max_outgoing) {
            max_outgoing = cur_neuron->synapses.size();
        }

        neuron_id_to_ind[cur_neuron->id] = i;
    }

    rnet->make_sorted_node_vector();
    input_ind_to_neuron_ind.resize(rnet->num_inputs());
    output_ind_to_neuron_ind.resize(rnet->num_outputs());
    for (i = 0; i < rnet->sorted_node_vector.size(); i++) {
        cur_node = rnet->sorted_node_vector[i];
        if (cur_node->is_input()) {
            input_ind_to_neuron_ind[cur_node->input_id] = i;
        }
        if (cur_node->is_output()) {
            output_ind_to_neuron_ind[cur_node->output_id] = i;
        }
    }

    s = "#define NUM_NEURONS (" + std::to_string(rnet->num_nodes()) +
        ")\n"
        "#define NUM_INPUT_NEURONS (" +
        std::to_string(rnet->num_inputs()) +
        ")\n"
        "#define NUM_OUTPUT_NEURONS (" +
        std::to_string(rnet->num_outputs()) +
        ")\n"
        "#define NUM_SYNAPSES (" +
        std::to_string(rnet->num_edges()) +
        ")\n"
        "#define MAX_NUM_TIMESTEPS (" +
        std::to_string(max_num_timesteps) +
        ")\n"
        "#define MAX_OUTGOING (" +
        std::to_string(max_outgoing) +
        ")\n"
        "#define MIN_POTENTIAL (" +
        std::to_string(min_potential) +
        ")\n"
        "#define SPIKE_VALUE_FACTOR (" +
        std::to_string(spike_value_factor) +
        ")\n"
        "\n"
        "unsigned long current_timestep = 0;\n"
        "\n"
        "const unsigned int INPUT_IND_TO_NEURON_IND[NUM_INPUT_NEURONS] = {";

    for (i = 0; i < input_ind_to_neuron_ind.size(); i++) {
        s += std::to_string(input_ind_to_neuron_ind[i]);
        if (i != input_ind_to_neuron_ind.size() - 1) {
            s += ", ";
        }
    }

    s += "};\nconst unsigned int OUTPUT_IND_TO_NEURON_IND[NUM_OUTPUT_NEURONS] "
         "= {";

    for (i = 0; i < output_ind_to_neuron_ind.size(); i++) {
        s += std::to_string(output_ind_to_neuron_ind[i]);
        if (i != output_ind_to_neuron_ind.size() - 1) {
            s += ", ";
        }
    }

    s += "};\n"
         "\n";

    if (fire_like_ravens) {
        fprintf(
            stderr,
            "EmbeddedRispSoA does not support the fire_like_ravens flag.\n");
        exit(1);
    }

    s += "unsigned char neuron_leak[NUM_NEURONS] = {\n    ";
    for (i = 0; i < enet->sorted_neuron_vector_public.size(); i++) {
        s += std::to_string(
            (unsigned char)enet->sorted_neuron_vector_public[i]->leak);
        if (i != enet->sorted_neuron_vector_public.size() - 1) {
            s += ", ";
        }
    }
    s += "\n};\n";

    s += "unsigned int neuron_outgoing[NUM_NEURONS] = {\n    ";
    for (i = 0; i < enet->sorted_neuron_vector_public.size(); i++) {
        s += std::to_string((unsigned int)enet->sorted_neuron_vector_public[i]
                                ->synapses.size());
        if (i != enet->sorted_neuron_vector_public.size() - 1) {
            s += ", ";
        }
    }
    s += "\n};\n";

    s += "unsigned int neuron_fire_count[NUM_NEURONS] = {0};\n";

    s += "int neuron_last_fire[NUM_NEURONS] = {0};\n";

    s += "unsigned int neuron_fire_times[NUM_NEURONS][MAX_NUM_TIMESTEPS] = "
         "{0};\n";

    s += "double neuron_charge_buffer[MAX_NUM_TIMESTEPS][NUM_NEURONS] = {0};\n";
    s += "unsigned char neuron_active[MAX_NUM_TIMESTEPS][NUM_NEURONS] = {0};\n";

    s += "double neuron_threshold[NUM_NEURONS] = {\n    ";
    for (i = 0; i < enet->sorted_neuron_vector_public.size(); i++) {
        s += std::to_string(
            (double)enet->sorted_neuron_vector_public[i]->threshold);
        if (i != enet->sorted_neuron_vector_public.size() - 1) {
            s += ", ";
        }
    }
    s += "\n};\n";

    s += "\n";

    s += "unsigned int synapse_to[NUM_NEURONS][MAX_OUTGOING] = {\n    ";
    for (i = 0; i < enet->sorted_neuron_vector_public.size(); i++) {
        risp::Neuron *node = enet->sorted_neuron_vector_public[i];

        s += "{";
        if (node->synapses.size() == 0) {
            s += "0";
        } else {
            for (j = 0; j < node->synapses.size(); j++) {
                s +=
                    std::to_string(neuron_id_to_ind[node->synapses[j]->to->id]);
                if (j != node->synapses.size() - 1) {
                    s += ", ";
                }
            }
        }
        s += "}";
        if (i != enet->sorted_neuron_vector_public.size() - 1) {
            s += ", ";
        }
    }
    s += "\n};\n";

    s += "unsigned int synapse_delay[NUM_NEURONS][MAX_OUTGOING] = {\n    ";
    for (i = 0; i < enet->sorted_neuron_vector_public.size(); i++) {
        risp::Neuron *node = enet->sorted_neuron_vector_public[i];

        s += "{";
        if (node->synapses.size() == 0) {
            s += "0";
        } else {
            for (j = 0; j < node->synapses.size(); j++) {
                s += std::to_string(node->synapses[j]->delay);
                if (j != node->synapses.size() - 1) {
                    s += ", ";
                }
            }
        }
        s += "}";
        if (i != enet->sorted_neuron_vector_public.size() - 1) {
            s += ", ";
        }
    }
    s += "\n};\n";

    s += "double synapse_weight[NUM_NEURONS][MAX_OUTGOING] = {\n    ";
    for (i = 0; i < enet->sorted_neuron_vector_public.size(); i++) {
        risp::Neuron *node = enet->sorted_neuron_vector_public[i];

        s += "{";
        if (node->synapses.size() == 0) {
            s += "0";
        } else {
            for (j = 0; j < node->synapses.size(); j++) {
                s += std::to_string((double)node->synapses[j]->weight);
                if (j != node->synapses.size() - 1) {
                    s += ", ";
                }
            }
        }
        s += "}";
        if (i != enet->sorted_neuron_vector_public.size() - 1) {
            s += ", ";
        }
    }
    s += "\n};\n";

    s += "\n";

    s += gen_apply_spike_c() + "\n";
    s += gen_run_c() + "\n";
    s += gen_clear_activity_c() + "\n";
    s += gen_output_last_fire_c() + "\n";
    s += gen_output_count_c();

    return s.get_str();
}

std::string EmbeddedRispSoA::gen_apply_spike_c() {
    IndentString s;

    s = "void apply_spike(unsigned int input_ind, unsigned int time, double "
        "value) {\n";

    s.add_indent_spaces(4);

    s += "unsigned int target_timestep;\n"
         "\n"
         "/* Ensure input neuron index is not out of bounds */\n"
         "if (input_ind >= NUM_INPUT_NEURONS) {\n"
         "    return;\n"
         "}\n"
         "\n"
         "/* Ensure time is not out of bounds */\n"
         "if (time >= MAX_NUM_TIMESTEPS) {\n"
         "    return;\n"
         "}\n"
         "\n"
         "target_timestep = (current_timestep + time) % "
         "MAX_NUM_TIMESTEPS;\n"
         "\n"
         "neuron_charge_buffer[target_timestep][INPUT_IND_TO_NEURON_IND[input_"
         "ind]] += value * SPIKE_VALUE_FACTOR;\n"
         "neuron_active[target_timestep][INPUT_IND_TO_NEURON_IND[input_"
         "ind]] = 1;\n";

    s.add_indent_spaces(-4);

    s += "}\n";

    return s.get_str();
}

std::string EmbeddedRispSoA::gen_run_c() {
    IndentString s;
    unsigned int i;
    bool net_all_leak;

    net_all_leak = true;
    for (i = 0; i < enet->sorted_neuron_vector_public.size(); i++) {
        if (!enet->sorted_neuron_vector_public[i]->leak) {
            net_all_leak = false;
            break;
        }
    }

    s = "void run(double duration) {\n";

    s.add_indent_spaces(4);

    s += "unsigned int time;\n"
         "unsigned int i;\n"
         "unsigned int j;\n"
         "unsigned int run_time;\n"
         "unsigned int cur_neuron_ind;\n"
         "unsigned int cur_synapse_ind;\n"
         "unsigned int to_time;\n"
         "\n"
         "/* Clear tracking info on all neurons */\n"
         "for (i = 0; i < NUM_NEURONS; i++) {\n"
         "    neuron_last_fire[i] = -1;\n"
         "    neuron_fire_count[i] = 0;\n"
         "}\n"
         "\n"
         "/* Ensure run_time is not negative */\n";

    if (run_time_inclusive) {
        s += "if (duration < 0) {\n"
             "    return;\n"
             "}\n"
             "\n"
             "run_time = (unsigned int)duration;\n";
    } else {
        s += "if (duration-1 < 0) {\n"
             "    return;\n"
             "}\n"
             "\n"
             "run_time = (unsigned int)(duration-1);\n";
    }

    s += "\n"
         "for (time = 0; time <= run_time; time++) {\n"
         "\n";

    s.add_indent_spaces(4);

    s += "unsigned int internal_timestep = (current_timestep + time) % "
         "MAX_NUM_TIMESTEPS;\n"
         "\n"
         "for (cur_neuron_ind = 0; cur_neuron_ind < NUM_NEURONS; "
         "cur_neuron_ind++) {\n";

    s.add_indent_spaces(4);

    // Min potential
    s += "if (neuron_charge_buffer[internal_timestep][cur_neuron_ind] < "
         "MIN_POTENTIAL) {\n";
    s.add_indent_spaces(4);
    s += "neuron_charge_buffer[internal_timestep][cur_neuron_ind] = "
         "MIN_POTENTIAL;\n";
    s.add_indent_spaces(-4);
    s += "}\n";

    // Did the neuron fire?
    if (threshold_inclusive) {
        s += "if (neuron_active[internal_timestep][cur_neuron_ind] && "
             "neuron_charge_buffer[internal_timestep][cur_neuron_ind] >= "
             "neuron_threshold[cur_neuron_ind]) {\n";
    } else {
        s += "if (neuron_active[internal_timestep][cur_neuron_ind] && "
             "neuron_charge_buffer[internal_timestep][cur_neuron_ind] > "
             "neuron_threshold[cur_neuron_ind]) {\n";
    }
    s.add_indent_spaces(4);

    s += "/* Neuron Fired, loop through synapses */\n";
    s += "for (cur_synapse_ind = 0; cur_synapse_ind < "
         "neuron_outgoing[cur_neuron_ind]; cur_synapse_ind++) {\n";
    s.add_indent_spaces(4);

    s += "neuron_charge_buffer[(internal_timestep + "
         "synapse_delay[cur_neuron_ind][cur_synapse_ind]) % "
         "MAX_NUM_TIMESTEPS][synapse_to[cur_neuron_ind][cur_synapse_ind]] += "
         "synapse_weight[cur_neuron_ind][cur_synapse_ind];\n";
    s += "neuron_active[(internal_timestep + "
         "synapse_delay[cur_neuron_ind][cur_synapse_ind]) % "
         "MAX_NUM_TIMESTEPS][synapse_to[cur_neuron_ind][cur_synapse_ind]] = "
         "1;\n";

    s.add_indent_spaces(-4);
    s += "}\n\n";

    // Output tracking
    s += "/* Ouptut tracking */\n";
    s += "neuron_fire_count[cur_neuron_ind]++;\n";
    s += "neuron_last_fire[cur_neuron_ind] = time;\n";
    s += "neuron_fire_times[cur_neuron_ind][neuron_fire_count[cur_neuron_ind]] "
         "= time;\n\n";

    // If all neurons leak there's no need to generate carry-over code
    if (!net_all_leak) {
        s.add_indent_spaces(-4);
        s += "} else {\n";
        s.add_indent_spaces(4);

        s += "/* Neuron did not fire, calculate carry-over */\n";
        s += "if (!neuron_leak[cur_neuron_ind]) {\n";
        s.add_indent_spaces(4);

        s += "neuron_charge_buffer[(internal_timestep + 1) % "
             "MAX_NUM_TIMESTEPS][cur_neuron_ind] += "
             "neuron_charge_buffer[internal_timestep][cur_neuron_ind];\n";

        s.add_indent_spaces(-4);
        s += "}\n\n";
    }

    s.add_indent_spaces(-4);
    s += "}\n\n";

    s.add_indent_spaces(-4);

    s += "}\n"
         "\n";

    // Loop through and clear row of matrix
    s += "/* Loop through and clear row of matrix (memset to 0) */\n";
    s += "for (cur_neuron_ind = 0; cur_neuron_ind < NUM_NEURONS; "
         "cur_neuron_ind++) {\n";
    s.add_indent_spaces(4);
    s += "neuron_charge_buffer[internal_timestep][cur_neuron_ind] = 0;\n";
    s += "neuron_active[internal_timestep][cur_neuron_ind] = 0;\n";
    s.add_indent_spaces(-4);
    s += "}\n";

    s.add_indent_spaces(-4);

    s += "}\n"
         "\n";

    if (run_time_inclusive) {
        s += "current_timestep += duration + 1;\n";
    } else {
        s += "current_timestep += duration;\n";
    }

    s.add_indent_spaces(-4);

    s += "}\n";

    return s.get_str();
}

std::string EmbeddedRispSoA::gen_clear_activity_c() {
    IndentString s;

    s = "void clear_activity() {\n";

    s.add_indent_spaces(4);

    s += "unsigned int i;\n"
         "unsigned int j;\n"
         "\n"
         "/* Clear activity-related neuron state */\n"
         "for (i = 0; i < NUM_NEURONS; i++) {\n"
         "    neuron_last_fire[i] = -1;\n"
         "    neuron_fire_count[i] = 0;\n"
         "}\n"
         "\n"
         "/* Clear all event activity */\n"
         "for (i = 0; i < MAX_NUM_TIMESTEPS; i++) {\n"
         "    for (j = 0; j < NUM_NEURONS; j++) {\n"
         "        neuron_charge_buffer[i][j] = 0;"
         "    }\n"
         "}\n"
         "for (i = 0; i < MAX_NUM_TIMESTEPS; i++) {\n"
         "    for (j = 0; j < NUM_NEURONS; j++) {\n"
         "        neuron_active[i][j] = 0;"
         "    }\n"
         "}\n";

    s.add_indent_spaces(-4);

    s += "}\n";

    return s.get_str();
}

std::string EmbeddedRispSoA::gen_output_last_fire_c() {
    IndentString s;

    s = "double output_last_fire(unsigned int output_ind) {\n";

    s.add_indent_spaces(4);

    s += "\n"
         "/* Ensure output index not out of bounds */\n"
         "if (output_ind >= NUM_OUTPUT_NEURONS) {\n"
         "    return -1;\n"
         "}\n"
         "\n"
         "return neuron_last_fire[OUTPUT_IND_TO_NEURON_IND[output_ind]];\n";

    s.add_indent_spaces(-4);

    s += "}\n";

    return s.get_str();
}

std::string EmbeddedRispSoA::gen_output_count_c() {
    IndentString s;

    s = "unsigned int output_count(unsigned int output_ind) {\n";

    s.add_indent_spaces(4);

    s += "\n"
         "/* Ensure output index not out of bounds */\n"
         "if (output_ind >= NUM_OUTPUT_NEURONS) {\n"
         "    return 0;\n"
         "}\n"
         "\n"
         "return neuron_fire_count[OUTPUT_IND_TO_NEURON_IND[output_ind]];\n";

    s.add_indent_spaces(-4);

    s += "}\n";

    return s.get_str();
}
