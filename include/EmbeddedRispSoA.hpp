#include "risp.hpp"

class EmbeddedRispSoA : public risp::Processor {

  protected:
    class EmbeddedRispSoANetwork : public risp::Network {
      public:
        EmbeddedRispSoANetwork(neuro::Network *net, double _spike_value_factor,
                               double _min_potential, char leak,
                               bool _run_time_inclusive,
                               bool _threshold_inclusive,
                               bool _fire_like_ravens, bool _discrete,
                               bool _inputs_from_weights, uint32_t _noisy_seed,
                               double _noisy_stddev, vector<double> &_weights,
                               vector<double> &_stds);

        std::vector<risp::Neuron *> sorted_neuron_vector_public;
    };

  public:
    EmbeddedRispSoA(neuro::json &params);
    ~EmbeddedRispSoA();

    bool load_network(neuro::Network *net, int network_id = 0);

    std::string gen_static_c(unsigned int max_num_timesteps);
    std::string gen_apply_spike_c();
    std::string gen_run_c();
    std::string gen_clear_activity_c();
    std::string gen_output_last_fire_c();
    std::string gen_output_count_c();

    neuro::Network *rnet;
    EmbeddedRispSoANetwork *enet;
};
