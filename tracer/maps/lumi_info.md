# General Architecture
LUMI is basically built out of 64-port switches. For LUMI-C and LUMI-G, 16 of those ports face compute nodes with the other 48 ports are used to make connections between switches, in a dragonfly topology.

The mapping between nodes and switches is not completely trivial and different from what you may be used to from other clusters, for two reasons:

- The hardware architecture: LUMI is built up from blades that contain either 4 CPU nodes or 2 GPU nodes, with network cards that provide each 2 endpoints. Two CPU nodes share a network card (but with individual endpoints), one GPU node has 2 network cards so 4 endpoints. On the GPU nodes, the network cards are connected to PCIe interfaces of the GPUs and not of the CPUs. These (vertical) blades then slot in (horizontal) switches. There are pictures in our course materials, including some in the first presentation of the 1-day course (https://lumi-supercomputer.github.io/intro-latest). So the 4 CPU nodes on a blade are distributed across two switches. I'm not 100% sure about the numbering, but I believe it is 0 and 1 on the first switch, then 2 and 3 on the next one, 4 and 5 (next blade) again on the first one, and so on, and not as is currently shown in the docs on https://docs.lumi-supercomputer.eu/hardware/network/. (The numbering on that picture is certainly not right as the numbering for LUMI-C starts from 1000 with node nid001000 not available and not visible in the scheduler). I'll also check with the person who I think wrote that documentation page. Likewise I think that the picture for the GPU nodes may also be wrong as I would expect that each node on a GPU blade is connected to a pair of switches rather than 4 (this would make cabling inside the node between the network cards and the ports to the switches a lot simpler). But in any case, each blade with 2 GPU nodes is connected to 4 switches. As each rack has only 62 or 63 blades rather than the full set of 64, there will be some irregularity in the mapping between node numbers and pairs or quads of switches.
- The network topology is a dragonfly topology with one group corresponding to one rack of up to 64 blades. Within a group there is an all-to-all connection between switches, and then there is also an all-to-all connection between groups. The logic is that the groups are kept small enough that all cabling can be copper which is cheaper than fibers, while the cabling between groups requires fibre.In principle inside a group each node can reach each other node via at most one hop between switches. However, assume that all nodes of your job would be on two switches then there is only one 200 Gbit/s link between those two switches which if communication is somewhat synchronized, would saturate. The flow control in the network will then reroute some messages over a longer path.Any node can in principle reach any other node in at most three hops between network switches, but again this is only theory, as long as there is no saturation of connections, after which messages or even parts of messages can be rerouted.On the 4-day course we have someone from Cray who is rather knowledgeable in the working of the network.

The smallest entity that can easily be selected on LUMI is not a switch but a group in the dragonfly topology. The scheduler does define a feature for that for each of the nodes in LUMI-C and LUMI-G.

# Xgroup, switches and IDS
There are  256 CPU nodes or 128 GPU nodes per cabinet (network group).

Detailed info will be found in the file `/etc/cray/xname`, that contains a code formatted as such:

- x#### = network group / cabinet
- c# = chassis (8 chassis / cabinet)
- s# = blade (8 blades / chassis)
- b# = board (2 boards / blade )
- n# = node (2 nodes / board (CPU), 1 node / board (GPU))
