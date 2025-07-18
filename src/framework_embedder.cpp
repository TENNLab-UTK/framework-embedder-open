#include "EmbeddedRisp.hpp"
#include "EmbeddedRispSoA.hpp"
#include "helpers.hpp"
#include "utils/cmdline.h"
#include "utils/json_helpers.hpp"
#include <iostream>

int main(int argc, char **argv) {
    std::string line;
    std::string j_str;
    std::string proc_name;
    std::string desired_processor;
    std::vector<std::string> net_data_keys;
    IndentString out_s;
    nlohmann::json j;
    neuro::Network net;
    EmbeddedRisp *emb_risp;
    EmbeddedRispSoA *emb_risp_soa;
    neuro::EdgeMap::iterator edge_it;
    cmdline::parser parse;
    double node_delay;
    int sim_time;
    int max_num_timesteps;

    try {

        // Load in command line arguments from user
        try {
            parse.add<string>(
                "processor", 'p',
                "which processor you'd like code to be generated for, "
                "<risp|rispSoA>",
                false, "risp", cmdline::oneof<string>("risp", "rispSoA"));

            parse.parse_check(argc, argv);

            desired_processor = parse.get<string>("processor");

        } catch (std::runtime_error &e) {
            std::cerr << "Error parsing command line arguments" << std::endl;
            throw e;
        }

        // Get json as a std::string from stdin (can also be a std::string
        // representing a file path to a json file)
        j_str = "";
        while (std::getline(std::cin, line)) {
            j_str += line;
        }

        // Read json from std::string or file
        try {
            j = json_from_string_or_file(j_str);
        } catch (std::runtime_error &e) {
            std::cerr << "Error reading the following json:\n"
                      << j << std::endl;
            throw e;
        } catch (const json::exception &e) {
            std::cerr << "Error reading the following json:\n"
                      << j << std::endl;
            throw e;
        }

        // Load json into corresponding TENNLab object via trial and error
        try {
            net.from_json(j);
        } catch (std::runtime_error &e) {
            throw(std::string) "Provided json cannot be parsed into a "
                                "network";
        }

        // Extract sim_time from network json if existant
        sim_time = -1;
        net_data_keys = net.data_keys();
        if (std::find(net_data_keys.begin(), net_data_keys.end(),
                        "other") != net_data_keys.end()) {
            try {
                j = net.get_data("other");
                if (j.contains("sim_time")) {
                    sim_time = j["sim_time"].get<int>();
                }
            } catch (...) {
                throw(std::string) "Error reading sim_time from the given "
                                    "network JSON's other associated data.";
            }
        }

        out_s = "";

        // Generate spiking neural network code for neuroprocessor

        // Extract processor parameters from network JSON
        try {
            j = net.get_data("proc_params");
        } catch (...) {
            throw(std::string) "Error reading proc_params from the given "
                                "network JSON's associated data.";
        }

        // Create risp processor object and load the given network
        if (desired_processor == "rispSoA") {
            emb_risp_soa = new EmbeddedRispSoA(j);
            emb_risp_soa->load_network(&net);
        } else {
            emb_risp = new EmbeddedRisp(j);
            emb_risp->load_network(&net);
        }

        // Determine the maximum number of timesteps to track in static C
        // code for the given network (either sim_time or maximum synapse
        // delay + 1)
        max_num_timesteps = -1;
        if (sim_time > 0) {
            max_num_timesteps = sim_time;
        }
        for (edge_it = net.edges_begin(); edge_it != net.edges_end();
                edge_it++) {
            node_delay = edge_it->second->get("Delay");
            if (node_delay + 1 > max_num_timesteps) {
                max_num_timesteps = node_delay + 1;
            }
        }
        if (max_num_timesteps < 2) {
            max_num_timesteps = 2;
        }

        if (desired_processor == "rispSoA") {
            // Write out neuroprocessor static C code
            out_s += emb_risp_soa->gen_static_c(max_num_timesteps) + "\n\n";

            delete emb_risp_soa;
        } else {
            // Write out neuroprocessor static C code
            out_s += emb_risp->gen_static_c(max_num_timesteps) + "\n\n";

            delete emb_risp;
        }

        std::cout << out_s.get_str();

    } catch (const json::exception &e) {
        std::cerr << e.what() << std::endl;
    } catch (std::runtime_error &e) {
        std::cerr << e.what() << std::endl;
    } catch (const std::string &e) {
        std::cerr << e << std::endl;
    }
}
