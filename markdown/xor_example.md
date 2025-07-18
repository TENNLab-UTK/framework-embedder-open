# XOR Example for Using the TENNLab Neuromorphic Framework Embedder

This tutorial for the TENNLab neuromorphic framework embedder tool (AKA the `framework_embedder`) covers an example XOR SNN model and how it may be converted into a simple C library.


------------------------------------------------------------

## Testing the XOR SNN using the open-source TENNLab neuromorphic framework

First, let's test out the provided RISP XOR model provided in the `networks` directory of this `framework_embedder` repo. We will begin by navigating to the TENNLab open-source framework repo:

```
UNIX> cd <path to TENNLab open source framework repo>
```

Let's go ahead and make all of the tools we'll need:

```
UNIX> make all
g++ -c -std=c++11 -Wall -Wextra -Iinclude -Iinclude/utils  -o obj/framework.o src/framework.cpp
g++ -c -std=c++11 -Wall -Wextra -Iinclude -Iinclude/utils  -o obj/processor_help.o src/processor_help.cpp
g++ -c -std=c++11 -Wall -Wextra -Iinclude -Iinclude/utils  -o obj/properties.o src/properties.cpp
ar r lib/libframework.a obj/framework.o obj/processor_help.o obj/properties.o
ar: creating lib/libframework.a
ranlib lib/libframework.a
g++ -std=c++11 -Wall -Wextra -Iinclude -Iinclude/utils  -o bin/network_tool src/network_tool.cpp lib/libframework.a
g++ -c -std=c++11 -Wall -Wextra -Iinclude -Iinclude/utils  -o obj/risp.o src/risp.cpp
g++ -c -std=c++11 -Wall -Wextra -Iinclude -Iinclude/utils  -o obj/risp_static.o src/risp_static.cpp
g++ -std=c++11 -Wall -Wextra -Iinclude -Iinclude/utils  -o bin/processor_tool_risp src/processor_tool.cpp obj/risp.o obj/risp_static.o lib/libframework.a
g++ -c -std=c++11 -Wall -Wextra -Iinclude -Iinclude/utils  -DNO_SIMD -o obj/vrisp.o src/vrisp.cpp
g++ -c -std=c++11 -Wall -Wextra -Iinclude -Iinclude/utils  -o obj/vrisp_static.o src/vrisp_static.cpp
g++ -std=c++11 -Wall -Wextra -Iinclude -Iinclude/utils  -o bin/processor_tool_vrisp src/processor_tool.cpp obj/vrisp.o obj/vrisp_static.o lib/libframework.a
g++ -std=c++11 -Wall -Wextra -Iinclude -Iinclude/utils  -o bin/network_to_jgraph src/network_to_jgraph.cpp lib/libframework.a
```

Now let's load in our XOR network into the `processor_tool`, which gives us an easy command line interface to test out the network with the regular TENNLab open-source framework simulator:

```
UNIX> ( echo ML <path to TENNLab framework embedder repo>/networks/xor.json ; echo INFO ) | bin/processor_tool_risp
Input nodes:  0 1 
Hidden nodes: 2 3 
Output nodes: 4 

```

Now let's test the XOR network's functionality. The `processor_tool` commands we'll use are apply spike (`AS node_id spike_time spike_val`), run (`RUN simulation_time`), and get all output neuron fire counts (`OC`). We'll start by testing 1^0=1:

```
UNIX> ( echo ML <path to TENNLab framework embedder repo>/networks/xor.json ; echo AS 0 0 1 ; echo RUN 3 ; echo OC ) | bin/processor_tool_risp
node 4 spike counts: 1
```

In this test, we applied a single input spike of magnitude 1.0 to the first input neuron (node_id = 0) at time 0 (i.e. the current timestep). This represents the first input bit of the XOR operation being 1. No spikes are applied to the second input neuron (node_id = 1), which represents the second input bit of the XOR operation being 0. We then run the SNN for 3 discrete timesteps. Finally, we check how many times our only output neuron fired. Since, the output node spiked 1 time, our output of 1^0 is equal to 1. Correct!

Next is 0^1=1:

```
UNIX> ( echo ML <path to TENNLab framework embedder repo>/networks/xor.json ; echo AS 1 0 1 ; echo RUN 3 ; echo OC ) | bin/processor_tool_risp
node 4 spike counts: 1
```

This test is exactly the same as the preceding one, except the sole input spike is applied to the second input neuron instead of the first one. This means the first input bit of the XOR operation is 0, and the second input bit of the XOR operation is 1. Our output node still spiked 1 time, meaning our output of 0^1 is equal to 1. Correct again!

Next is 1^1=0:

```
UNIX> ( echo ML <path to TENNLab framework embedder repo>/networks/xor.json ; echo AS 0 0 1 ; echo AS 1 0 1 ; echo RUN 3 ; echo OC ) | bin/processor_tool_risp
node 4 spike counts: 0
```

Now we apply two input spikes to the XOR SNN: one to the first input neuron and one to the second input neuron. Note that we need to ensure that we apply these two input spikes at the same timestep to get our desired output. Now our output node spiked 0 times, meaning the SNN calculated 1^1=0. Correct again!

Lastly is 0^0=0:

```
UNIX> ( echo ML <path to TENNLab framework embedder repo>/networks/xor.json ; echo RUN 3 ; echo OC ) | bin/processor_tool_risp
node 4 spike counts: 0
```

Now we apply zero input spikes to the XOR SNN. Now our output node spiked 0 times, meaning the SNN calculated 1^1=0. Correct again! Note that this is the same as just not running the SNN at all since no input activity is introduced.


------------------------------------------------------------

## Using the `framework_embedder` to generate a simple C library from the XOR SNN

Let's make sure we're back in this open-source `framework_embedder` repo:

```
UNIX> cd <path to open-source framework embedder repo>
```

The open-source `framework_embedder` depends on the TENNLab open-source framework repo, so we need to give the makefile the path to the framework. We can either manually set this path in the makefile itself or just set the `fr_open` environment variable. For the sake of this tutorial, we'll just set the environment variable:

```
UNIX> export fr_open=<path to TENNLab open source framework repo>
```

Note that this environment variable will not still be set if we reset our machine.

Now we can compile the `framework_embedder` tool:

```
UNIX> make all
( mkdir bin ; cd <path to TENNLab open source framework repo> ; make )
mkdir: cannot create directory ‘bin’: File exists
make[1]: Entering directory '<path to TENNLab open source framework repo>'
make[1]: Nothing to be done for 'all'.
make[1]: Leaving directory '<path to TENNLab open source framework repo>'
g++ -o bin/framework_embedder -Wall -Wextra -std=c++11 -I<path to TENNLab open source framework repo>/include -I./include/ src/*.cpp <path to TENNLab open source framework repo>/obj/framework.o <path to TENNLab open source framework repo>/obj/risp.o <path to TENNLab open source framework repo>/obj/risp_static.o <path to TENNLab open source framework repo>/lib/libframework.a
```

Now, let's compile our XOR SNN model from its JSON representation into a simple C header file:

```
UNIX> mkdir tmp_xor_example ; bin/framework_embedder < networks/xor.json > tmp_xor_example/xor.h
```

And that's it! We now have a dependency-free, lightweight C library in `tmp_xor_example/xor.h`, whose sole purpose is to simulate just the XOR SNN model. Notice that it's just a single file!

`tmp_xor_example/xor.h` can be compiled for any piece of hardware that has a C compiler+toolchain (e.g. a microcontroller). For the sake of this tutorial, we can now make a simple main program to test out `tmp_xor_example/xor.h` on our own machine:

```
UNIX> echo "#include \"xor.h\"" > tmp_xor_example/xor_test.c ; \
echo "#include <stdio.h>" >> tmp_xor_example/xor_test.c ; \
echo "int main() {" >> tmp_xor_example/xor_test.c ; \
echo "  clear_activity();                    /* Clear SNN activity */" >> tmp_xor_example/xor_test.c ; \
echo "  apply_spike(0, 0, 1);                /* Apply spike of magnitude 1.0 to first input neuron at timestep 0 (current timestep) */" >> tmp_xor_example/xor_test.c ; \
echo "  run(3);                              /* Run SNN for 3 discrete timesteps */" >> tmp_xor_example/xor_test.c ; \
echo "  printf(\"1^0=%d\\n\",output_count(0));  /* Get the number of times that the first and only output neuron fired */" >> tmp_xor_example/xor_test.c ; \
echo "  apply_spike(1, 0, 1);                /* Apply spike of magnitude 1.0 to second input neuron at timestep 0 (current timestep) */" >> tmp_xor_example/xor_test.c ; \
echo "  run(3);                              /* Run SNN for 3 discrete timesteps */" >> tmp_xor_example/xor_test.c ; \
echo "  printf(\"0^1=%d\\n\",output_count(0));  /* Get the number of times that the first and only output neuron fired */" >> tmp_xor_example/xor_test.c ; \
echo "  apply_spike(0, 0, 1);                /* Apply spike of magnitude 1.0 to first input neuron at timestep 0 (current timestep) */" >> tmp_xor_example/xor_test.c ; \
echo "  apply_spike(1, 0, 1);                /* Apply spike of magnitude 1.0 to second input neuron at timestep 0 (current timestep) */" >> tmp_xor_example/xor_test.c ; \
echo "  run(3);                              /* Run SNN for 3 discrete timesteps */" >> tmp_xor_example/xor_test.c ; \
echo "  printf(\"1^1=%d\\n\",output_count(0));  /* Get the number of times that the first and only output neuron fired */" >> tmp_xor_example/xor_test.c ; \
echo "  run(3);                              /* Run SNN for 3 discrete timesteps */" >> tmp_xor_example/xor_test.c ; \
echo "  printf(\"0^0=%d\\n\",output_count(0));  /* Get the number of times that the first and only output neuron fired */" >> tmp_xor_example/xor_test.c ; \
echo "  return 0;" >> tmp_xor_example/xor_test.c ; \
echo "}" >> tmp_xor_example/xor_test.c
```

Let's take a quick look at the main program we just wrote:

```
UNIX> cat tmp_xor_example/xor_test.c
#include "xor.h"
#include <stdio.h>
int main() {
  clear_activity(); /* Clear SNN activity */
  apply_spike(0, 0, 1);            /* Apply spike of magnitude 1.0 to first input neuron at timestep 0 (current timestep) */
  run(3);                          /* Run SNN for 3 discrete timesteps */
  printf(1^0=%d,output_count(0));  /* Get the number of times that the first and only output neuron fired */
  apply_spike(1, 0, 1);            /* Apply spike of magnitude 1.0 to second input neuron at timestep 0 (current timestep) */
  run(3);                          /* Run SNN for 3 discrete timesteps */
  printf(0^1=%d,output_count(0));  /* Get the number of times that the first and only output neuron fired */
  apply_spike(0, 0, 1);            /* Apply spike of magnitude 1.0 to first input neuron at timestep 0 (current timestep) */
  apply_spike(1, 0, 1);            /* Apply spike of magnitude 1.0 to second input neuron at timestep 0 (current timestep) */
  run(3);                          /* Run SNN for 3 discrete timesteps */
  printf(1^1=%d,output_count(0));  /* Get the number of times that the first and only output neuron fired */
  run(3);                          /* Run SNN for 3 discrete timesteps */
  printf(0^0=%d,output_count(0));  /* Get the number of times that the first and only output neuron fired */
  return 0;
}
```

Now we can compile our XOR test program:

```
UNIX> gcc tmp_xor_example/xor_test.c -o tmp_xor_example/xor_test
```

And now we can finally run our XOR test program:

```
UNIX> tmp_xor_example/xor_test
1^0=1
0^1=1
1^1=0
0^0=0
```

Success! Everything works as expected, just like it did when testing the SNN using the TENNLab open-source framework's `processor_tool` in the section above.
