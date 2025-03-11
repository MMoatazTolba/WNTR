"""
WNTR Simulator thermal model
The meshnet function splits pipes based on the given maximum_pipe_length option under wntr.network.options.thermal
"""
import copy
import numpy as np
import pandas as pd
from wntr.network.model import WaterNetworkModel, NodeRegistry, Pipe
from wntr.sim.results import SimulationResults
from wntr.morph import split_pipe


class ThermalSimulator:
    """The ThermalSimulator class employs the upwind finite-volume dicretization method to solve the energy equation in 
       Water Distribution networks. """
    
    def __init__(self, wn: WaterNetworkModel, hyd_sim_results: SimulationResults):
        
        self._hyd_sim_results = hyd_sim_results
        
        self._timestep = wn.options.time.hydraulic_timestep 
        self._duration = wn.options.time.duration
        self._times = np.arange(0, self._duration + self._timestep, self._timestep, dtype='int64')
        self._timesteps_num = len(self._times)
        self._time_ids = np.arange(self._timesteps_num)
        
        self._nodes_names = wn.junction_name_list + wn.tank_name_list + wn.reservoir_name_list 
        self._nodes_num = wn.num_nodes
        self._nodes_ids = np.arange(self._nodes_num)
        
        self._links_names = wn.pipe_name_list + wn.head_pump_name_list + wn.power_pump_name_list + wn.valve_name_list 
        self._links_num= wn.num_links
        self._links_ids = np.arange(self._links_num)
        
        self._nodes_name2id = {k:v for (k,v) in zip(self._nodes_names, self._nodes_ids)}
        self._links_name2id = {k:v for (k,v) in zip(self._links_names, self._links_ids)}
        
        self._nodes = NodeRegistry(wn) 
        for node_name in self._nodes_names:
            # 1- Fill the new NodeRegistery with the model's nodes but in the same order as in the columns of results.node object
            # 2- Create new properties: _connected_links_id list and the _neighbour_nodes_id list that use ids instead of names
            self._nodes._data[node_name] = wn.nodes[node_name]
            self._nodes[node_name]._id = self._nodes_name2id[node_name]
            self._nodes[node_name]._connected_links_ids = np.array( [self._links_name2id[i] for i in self._nodes[node_name]._connected_links] )
            self._nodes[node_name]._neighbour_nodes_ids = np.array( [self._nodes_name2id[i] for i in self._nodes[node_name]._neighbour_nodes] )
        
        for link_name in self._links_names:
            wn.links[link_name]._id = self._links_name2id[link_name]
        
        self._junctions_names = wn.junction_name_list 
        self._junctions_num = wn.num_junctions
        self._junctions_ids = self._nodes_ids[0:self._junctions_num]
        self._junctions = wn.junctions
        
        self._tanks_names =  wn.tank_name_list 
        self._tanks_num = wn.num_tanks
        self._tanks_ids = self._nodes_ids[self._junctions_num : self._junctions_num + self._tanks_num]
        self._tanks = wn.tanks
        
        self._reservoirs_names = wn.reservoir_name_list 
        self._reservoirs_num = wn.num_reservoirs
        self._reservoirs_ids = self._nodes_ids[-self._reservoirs_num:]
        self._reservoirs = wn.reservoirs
        
        self._fluid_density = wn.options.hydraulic.specific_gravity * 1000
        self._fluid_viscosity = wn.options.hydraulic.viscosity
        self._fluid_heat_capacity = wn.options.thermal.heat_capacity
        
        self._flow_val = np.abs(self._hyd_sim_results.link['flowrate'].values) 
        self._flow_dir = np.sign(self._hyd_sim_results.link['flowrate'].values) 
        self._demands = self._hyd_sim_results.node['demand'].values 
        self._tanks_levels = self._hyd_sim_results.node['head'].values[:,self._tanks_ids] - np.array( [self._nodes[tank_name].elevation for tank_name in self._tanks_names] )
        
        self._temperatures = np.zeros((self._timesteps_num, self._nodes_num))
        self._tanks_volumes = np.zeros((self._timesteps_num, self._tanks_num))
        self._coef_matrix = None
        self._result_vector = None
        
    
    def _initialize_matrices(self):
        self._temperatures[0,:-self._reservoirs_num] = [self._nodes[node_name]._initial_temperature for node_name in self._junctions_names+self._tanks_names]
        self._temperatures[0,-self._reservoirs_num:] = [self._nodes[node_name].temperature_at(0) for node_name in self._reservoirs_names] 
        self._tanks_volumes[0,:] = [self._nodes[tank_name].get_volume(self._nodes[tank_name].init_level) for tank_name in self._tanks_names]
            
    
    def _build_linear_system(self, t, node, acc_volume = 0):
        in_out = self._flow_dir[t, node._connected_links_ids] * node._connection_side
        
        upstream_links_ids = node._connected_links_ids[in_out > 0] 
        upstream_nodes_ids = node._neighbour_nodes_ids[in_out > 0]
        downstream_links_ids = node._connected_links_ids[in_out <= 0]
        
        rho_cp_R_inv = node.total_thermal_resistance_reciprocal / (self._fluid_density * self._fluid_heat_capacity)
        
        self._coef_matrix[node._id, node._id] = (node._cell_volume + acc_volume)/ self._timestep + self._flow_val[t, downstream_links_ids].sum() + self._demands[t, node._id] + rho_cp_R_inv                                                  
        self._coef_matrix[node._id, upstream_nodes_ids] = -self._flow_val[t, upstream_links_ids] 
        self._result_vector[node._id] = node.soil_temperature_at(self._times[t]) * rho_cp_R_inv  + self._temperatures[t-1, node._id] * (node._cell_volume + acc_volume)/ self._timestep
        
        
    def run_sim(self):
        self._initialize_matrices()
        
        for t in self._time_ids[1:]:
            self._temperatures[t,-self._reservoirs_num:] = [self._nodes[node_name].temperature_at(self._times[t]) for node_name in self._reservoirs_names] 
            self._tanks_volumes[t,:] = [self._nodes[self._nodes_names[tank_id]].get_volume(self._tanks_levels[t, tank_id-self._junctions_num]) for tank_id in self._tanks_ids]
            
            self._coef_matrix = np.identity(self._nodes_num)
            self._result_vector = np.zeros(self._nodes_num)
            self._result_vector[-self._reservoirs_num:] = self._temperatures[t,-self._reservoirs_num:]
            
            for junction_name in self._junctions_names:
                self._build_linear_system(t, self._nodes[junction_name])
                
            for tank_name in self._tanks_names:
                self._build_linear_system(t, self._nodes[tank_name], self._tanks_volumes[self._times[t], self._nodes[tank_name]._id-self._junctions_num])

            self._temperatures[t,:] = np.linalg.solve(self._coef_matrix, self._result_vector)
        return pd.DataFrame(self._temperatures, index=self._times, columns= self._nodes_names)



def meshnet(wn: WaterNetworkModel, return_copy: bool=True) -> WaterNetworkModel:
    """Subdivide network pipes based on defined maximum pipe length"""

    if return_copy:
        meshed_wn = copy.deepcopy(wn)
    else:
        meshed_wn = wn

    if wn.options.thermal.max_pipe_length is None:
        print("\nThe meshnet function is called while the maximum pipe length is set to None! " 
              "The code will continue without meshing. \n"
              "If you wish to mesh the network, set a value for wn.network.options.thremal.max_pipe_length that is at"
              "most less than the longest pipe the network, where wn is of a type thermonet.WaterNetworkModel. \n \n")

    else:
        long_pipes = wn.query_link_attribute(attribute='length',
                                             operation=np.greater,
                                             value=wn.options.thermal.max_pipe_length,
                                             link_type=Pipe)
        
        if len(long_pipes) == 0:
            print("The maximum pipe length is set to a value larger than the longest pipe in the network. No meshing will take place. If you wish to mesh the network, set a value for wn.network.options.thremal.max_pipe_length that is at most less than the longest pipe the network, where wn is of a type thermonet.WaterNetworkModel")
        
        else:

            num_splits = long_pipes.floordiv(wn.options.thermal.max_pipe_length).astype(int)
    
            for pipe_name in long_pipes.index:
                
    
                for i in range(num_splits[pipe_name]):
    
                    ratio = 1 - 1 / (num_splits[pipe_name] + 1 - i)
    
                    num = (num_splits[pipe_name] - i)
                    new_pipe_name = f'{pipe_name}_{num}'
                    new_junction_name = f'of_{pipe_name}_{num}'
                    
                    meshed_wn = split_pipe(meshed_wn, pipe_name, new_pipe_name, new_junction_name,
                                    split_at_point=ratio, return_copy=False)


    return meshed_wn