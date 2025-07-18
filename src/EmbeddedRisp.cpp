#include "EmbeddedRisp.hpp"
#include "helpers.hpp"
#include <unordered_map>

EmbeddedRisp::EmbeddedRispNetwork::EmbeddedRispNetwork(
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

EmbeddedRisp::EmbeddedRisp(neuro::json &params) : risp::Processor(params) {
    rnet = nullptr;
    enet = nullptr;
}

EmbeddedRisp::~EmbeddedRisp() {
    if (enet != nullptr) {
        delete enet;
    }
}

bool EmbeddedRisp::load_network(neuro::Network *net, int network_id) {
    bool res;

    res = risp::Processor::load_network(net, network_id);

    rnet = net;
    enet = new EmbeddedRispNetwork(
        net, spike_value_factor, min_potential, leak_mode[0],
        run_time_inclusive, threshold_inclusive, fire_like_ravens, discrete,
        inputs_from_weights, noisy_seed, noisy_stddev, weights, stds);

    return res;
}

std::string EmbeddedRisp::gen_static_c(unsigned int max_num_timesteps) {
    IndentString s;
    risp::Neuron *cur_neuron;
    neuro::Node *cur_node;
    risp::Synapse *cur_synapse;
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

    s = "/******************* RISP NETWORK CODE ***********************/\n"
        "\n"
        "#define NUM_NEURONS (" +
        std::to_string(rnet->num_nodes()) +
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
        "/* Synapse struct */\n"
        "typedef struct {\n"
        "    unsigned int to;    /* Index of to neuron */\n"
        "    unsigned int delay; /* Synapse delay value */\n"
        "    double weight;       /* Synapse weight value */\n"
        "} Synapse;\n"
        "\n"
        "/* Neuron struct */\n"
        "typedef struct {\n"
        "    unsigned char leak;                         /* Leak value "
        "(1 for full leak and 0 for no leak) */\n"
        "    unsigned char check;                        /* Whether or "
        "not we have checked if this neuron fires */\n"
        "    unsigned int num_outgoing;                  /* Number of "
        "outgoing synapses for this neuron */\n"
        "    unsigned int fire_count;                    /* Number of "
        "fires */\n"
        "    int last_fire;                              /* Last firing "
        "time */\n"
        "    double charge;                               /* Charge "
        "value */\n"
        "    double threshold;                            /* Threshold "
        "value */\n"
        "    Synapse outgoing[MAX_OUTGOING];             /* Outgoing "
        "synapses */\n"
        "    unsigned int fire_times[MAX_NUM_TIMESTEPS]; /* Firing "
        "times */\n"
        "} Neuron;\n"
        "\n"
        "/* Charge change event struct (essentially just a pair) */\n"
        "typedef struct {\n"
        "    unsigned int neuron_ind; /* Index of neuron to change the "
        "charge for */\n"
        "    double charge_change;     /* Value to change charge by */\n"
        "} Charge_Change_Event;\n"
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
         "\n"
         "unsigned int event_count[MAX_NUM_TIMESTEPS] = {0};                   "
         "/* Number of charge change events for each timestep */\n"
         "unsigned int cur_charge_changes_ind = 0;                             "
         "/* Index of charge changes array that represents which array of "
         "charge change events corresponds to the upcoming timestep */\n"
         "Charge_Change_Event charge_changes[MAX_NUM_TIMESTEPS][NUM_SYNAPSES]; "
         "/* Charge changes keyed on timestep and charge change event index "
         "*/\n";

    if (fire_like_ravens) {
        s += "unsigned int to_fire[NUM_NEURONS]; /* Neuron indices for "
             "neurons that need to be fired at the beginning of the upcoming "
             "timestep */\n"
             "unsigned int to_fire_count = 0;    /* Number of neurons that "
             "need to be fired at the beginning of the upcoming timestep */\n";
    }

    s += "Neuron neurons[NUM_NEURONS] = { ";
    for (i = 0; i < enet->sorted_neuron_vector_public.size(); i++) {
        cur_neuron = enet->sorted_neuron_vector_public[i];

        s += "{" + std::to_string((unsigned char)cur_neuron->leak) + ", 0, " +
             std::to_string(cur_neuron->synapses.size()) + ", 0, -1, 0, " +
             std::to_string(cur_neuron->threshold) + ", {";

        for (j = 0; j < cur_neuron->synapses.size(); j++) {
            cur_synapse = cur_neuron->synapses[j];
            s += "{" + std::to_string(neuron_id_to_ind[cur_synapse->to->id]) +
                 "," + std::to_string(cur_synapse->delay) + "," +
                 std::to_string(cur_synapse->weight) + "}";
            if (j != cur_neuron->synapses.size() - 1) {
                s += ", ";
            }
        }

        if (cur_neuron->synapses.size() == 0) {
            s += "{0}";
        }

        s += "}, {0}}";

        if (i != enet->sorted_neuron_vector_public.size() - 1) {
            s += ",\n                                ";
        }
    }
    s += " };\n"
         "\n";

    s += gen_apply_spike_c() + "\n";
    s += gen_run_c() + "\n";
    s += gen_clear_activity_c() + "\n";
    s += gen_output_last_fire_c() + "\n";
    s += gen_output_count_c();

    return s.get_str();
}

std::string EmbeddedRisp::gen_apply_spike_c() {
    IndentString s;

    s = "/* This function will apply a spike of potential value value to the "
        "input neuron with an input neuron zero-based index of input_ind at "
        "time time relative to the current timestep of the neuroprocessor. */\n"
        "void apply_spike(unsigned int input_ind, unsigned int time, double "
        "value) {\n";

    s.add_indent_spaces(4);

    s += "unsigned int target_charge_changes_ind;\n"
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
         "target_charge_changes_ind = (cur_charge_changes_ind + time) % "
         "MAX_NUM_TIMESTEPS;\n"
         "\n"
         "/* Schedule charge change at current timestep for neuron with given "
         "index */\n"
         "if(event_count[target_charge_changes_ind] < NUM_SYNAPSES) {\n"
         "    "
         "charge_changes[target_charge_changes_ind][event_count[target_charge_"
         "changes_ind]].neuron_ind = INPUT_IND_TO_NEURON_IND[input_ind];\n"
         "    "
         "charge_changes[target_charge_changes_ind][event_count[target_charge_"
         "changes_ind]].charge_change = value * SPIKE_VALUE_FACTOR;\n"
         "    event_count[target_charge_changes_ind]++;\n"
         "}\n";

    s.add_indent_spaces(-4);

    s += "}\n";

    return s.get_str();
}

std::string EmbeddedRisp::gen_run_c() {
    IndentString s;
    unsigned int i;
    bool net_has_leak;

    net_has_leak = false;
    for (i = 0; i < enet->sorted_neuron_vector_public.size(); i++) {
        if (enet->sorted_neuron_vector_public[i]->leak) {
            net_has_leak = true;
        }
    }

    s = "/* This function will run the SNN for duration, the specified number "
        "of timesteps (many neuroprocessors only support discrete timesteps, "
        "such as RISP). */\n"
        "void run(double duration) {\n";

    s.add_indent_spaces(4);

    s += "unsigned int time;\n"
         "unsigned int i;\n"
         "unsigned int j;\n"
         "unsigned int run_time;\n"
         "unsigned int cur_neuron_ind;\n"
         "unsigned int to_time;\n"
         "\n"
         "/* Clear tracking info on all neurons */\n"
         "for (i = 0; i < NUM_NEURONS; i++) {\n"
         "    neurons[i].last_fire = -1;\n"
         "    neurons[i].fire_count = 0;\n"
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

    if (fire_like_ravens) {
        s += "/* Cause any neuron to fire the timestep after its charge "
             "exceeds its threshold (like RAVENS) */\n"
             "for (i = 0; i < to_fire_count; i++) {\n"
             "    cur_neuron_ind = to_fire[i];\n"
             "    "
             "neurons[cur_neuron_ind].fire_times[neurons[cur_neuron_ind].fire_"
             "count] = time;\n"
             "    neurons[cur_neuron_ind].last_fire = time;\n"
             "    neurons[cur_neuron_ind].fire_count++;\n"
             "    neurons[cur_neuron_ind].charge = 0;\n"
             "}\n"
             "to_fire_count = 0;\n"
             "\n";
    }

    if (net_has_leak) {
        s += "/* Apply leak and reset minimum charge before the next run */\n";
    } else {
        s += "/* Reset minimum charge before the next run */\n";
    }

    s += "for (i = 0; i < event_count[cur_charge_changes_ind]; i++) {\n";

    s.add_indent_spaces(4);

    s += "cur_neuron_ind = "
         "charge_changes[cur_charge_changes_ind][i].neuron_ind;\n";

    if (net_has_leak) {
        s += "if (neurons[cur_neuron_ind].leak) {\n"
             "    neurons[cur_neuron_ind].charge = 0;\n"
             "}\n";
    }

    s += "if (neurons[cur_neuron_ind].charge < MIN_POTENTIAL) {\n"
         "    neurons[cur_neuron_ind].charge = MIN_POTENTIAL;\n"
         "}\n";

    s.add_indent_spaces(-4);

    s += "}\n"
         "\n";

    s += "/* Collect charges */\n"
         "for (i = 0; i < event_count[cur_charge_changes_ind]; i++) {\n"
         "    cur_neuron_ind = "
         "charge_changes[cur_charge_changes_ind][i].neuron_ind;\n"
         "    neurons[cur_neuron_ind].check = 1;\n"
         "    neurons[cur_neuron_ind].charge += "
         "charge_changes[cur_charge_changes_ind][i].charge_change;\n"
         "}\n"
         "\n"
         "/* Determine if neuron fires */\n"
         "for (i = 0; i < event_count[cur_charge_changes_ind]; i++) {\n";

    s.add_indent_spaces(4);

    s += "cur_neuron_ind = "
         "charge_changes[cur_charge_changes_ind][i].neuron_ind;\n"
         "\n"
         "if (neurons[cur_neuron_ind].check == 1) {\n";

    s.add_indent_spaces(4);

    if (threshold_inclusive) {
        s += "/* Fire if neuron charge meets its threshold */\n"
             "if (neurons[cur_neuron_ind].charge >= "
             "neurons[cur_neuron_ind].threshold) {\n";
    } else {
        s += "/* Fire if neuron charge exceeds its threshold */\n"
             "if (neurons[cur_neuron_ind].charge > "
             "neurons[cur_neuron_ind].threshold) {\n";
    }

    s.add_indent_spaces(4);

    s += "for (j = 0; j < neurons[cur_neuron_ind].num_outgoing; j++) {\n"
         "    to_time = (cur_charge_changes_ind + "
         "neurons[cur_neuron_ind].outgoing[j].delay) % MAX_NUM_TIMESTEPS;\n"
         "    if (event_count[to_time] < NUM_SYNAPSES) {\n"
         "        charge_changes[to_time][event_count[to_time]].neuron_ind = "
         "neurons[cur_neuron_ind].outgoing[j].to;\n"
         "        charge_changes[to_time][event_count[to_time]].charge_change "
         "= neurons[cur_neuron_ind].outgoing[j].weight;\n"
         "        event_count[to_time]++;\n"
         "    }\n"
         "}\n"
         "\n";

    if (fire_like_ravens) {
        s += "to_fire[to_fire_count] = cur_neuron_ind;\n"
             "to_fire_count++;\n";
    } else {
        s += "neurons[cur_neuron_ind].fire_times[neurons[cur_neuron_ind].fire_"
             "count] = time;\n"
             "neurons[cur_neuron_ind].last_fire = time;\n"
             "neurons[cur_neuron_ind].fire_count++;\n"
             "neurons[cur_neuron_ind].charge = 0;\n";
    }

    s.add_indent_spaces(-4);

    s += "}\n"
         "\n"
         "neurons[cur_neuron_ind].check = 0;\n";

    s.add_indent_spaces(-4);

    s += "}\n";

    s.add_indent_spaces(-4);

    s += "}\n"
         "\n"
         "/* \"Shift\" (using ring buffer) extra spiking events up a timestep "
         "to progress to the next timestep */\n"
         "event_count[cur_charge_changes_ind] = 0;\n"
         "cur_charge_changes_ind = (cur_charge_changes_ind + 1) % "
         "MAX_NUM_TIMESTEPS;\n";

    s.add_indent_spaces(-4);

    s += "}\n"
         "\n";

    if (net_has_leak) {
        s += "/* Apply leak and reset minimum charge before the next run */\n";
    } else {
        s += "/* Reset minimum charge before the next run */\n";
    }

    s += "for (i = 0; i < NUM_NEURONS; i++) {\n";

    s.add_indent_spaces(4);

    if (net_has_leak) {
        s += "if (neurons[i].leak == 1) {\n"
             "    neurons[i].charge = 0;\n"
             "}\n";
    }

    s += "if (neurons[i].charge < MIN_POTENTIAL) {\n"
         "    neurons[i].charge = MIN_POTENTIAL;\n"
         "}\n";

    s.add_indent_spaces(-4);

    s += "}\n";

    s.add_indent_spaces(-4);

    s += "}\n";

    return s.get_str();
}

std::string EmbeddedRisp::gen_clear_activity_c() {
    IndentString s;

    s = "/* This function will clear the SNN of all activity. It resets all "
        "neuron and synapse state. */\n"
        "void clear_activity() {\n";

    s.add_indent_spaces(4);

    s += "unsigned int i;\n"
         "\n"
         "/* Clear activity-related neuron state */\n"
         "for (i = 0; i < NUM_NEURONS; i++) {\n"
         "    neurons[i].last_fire = -1;\n"
         "    neurons[i].fire_count = 0;\n"
         "    neurons[i].charge = 0;\n"
         "}\n"
         "\n"
         "/* Clear all event activity */\n"
         "for (i = 0; i < MAX_NUM_TIMESTEPS; i++) {\n"
         "    event_count[i] = 0;\n"
         "}\n";

    if (fire_like_ravens) {
        s += "\n"
             "/* Clear scheduled neuron fires */\n"
             "to_fire_count = 0;\n";
    }

    s.add_indent_spaces(-4);

    s += "}\n";

    return s.get_str();
}

std::string EmbeddedRisp::gen_output_last_fire_c() {
    IndentString s;

    s = "/* This function will return the timestep of the output neuron with "
        "an output neuron zero-based index of output_ind. The returned "
        "timestep will only be for the most recent call of the run() function. "
        "*/\n"
        "double output_last_fire(unsigned int output_ind) {\n";

    s.add_indent_spaces(4);

    s += "\n"
         "/* Ensure output index not out of bounds */\n"
         "if (output_ind >= NUM_OUTPUT_NEURONS) {\n"
         "    return -1;\n"
         "}\n"
         "\n"
         "return "
         "(double)neurons[OUTPUT_IND_TO_NEURON_IND[output_ind]].last_fire;\n";

    s.add_indent_spaces(-4);

    s += "}\n";

    return s.get_str();
}

std::string EmbeddedRisp::gen_output_count_c() {
    IndentString s;

    s = "/* This function will return the number of neuronal fires for the "
        "output neuron with an output neuron zero-based index of output_ind. "
        "The returned fire count will only be for the most recent call of the "
        "run() function. */\n"
        "unsigned int output_count(unsigned int output_ind) {\n";

    s.add_indent_spaces(4);

    s += "\n"
         "/* Ensure output index not out of bounds */\n"
         "if (output_ind >= NUM_OUTPUT_NEURONS) {\n"
         "    return 0;\n"
         "}\n"
         "\n"
         "return neurons[OUTPUT_IND_TO_NEURON_IND[output_ind]].fire_count;\n";

    s.add_indent_spaces(-4);

    s += "}\n";

    return s.get_str();
}
