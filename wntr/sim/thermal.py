"""
WNTR Simulator thermal model
TimeData a structure that holds time data together
ObjData a structures that holds common data for network objects together
HydraulicData a structure that holds the hydraulic data relevant for the thermal model together
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
       Water Distribution networks. 
       
       Parameters
       ----------
       
       wn : :class:`~wntr.network.model.WaterNetworkModel`
           The water network model.
       hyd_sim_results : :class:`~wntr.network.results.SimulationResults`
           The object containing results from the hydraulic simulation.
           Only flowrates, demands, and tanks' heads are used"""
    
    def __init__(self, wn: WaterNetworkModel, hyd_sim_results: SimulationResults):
        
        self._model = wn
        
        self._time = TimeData(wn.options.time.hydraulic_timestep, wn.options.time.duration)
        
        self._nodes_data      = ObjData(wn.junction_name_list + wn.tank_name_list + wn.reservoir_name_list , np.arange(wn.num_nodes))
        self._links_data      = ObjData(wn.pipe_name_list + wn.head_pump_name_list + wn.power_pump_name_list + wn.valve_name_list , np.arange(wn.num_links))
        self._junctions_data  = ObjData(wn.junction_name_list, self._nodes_data.ids[0:wn.num_junctions])   
        self._tanks_data      = ObjData(wn.tank_name_list, self._nodes_data.ids[wn.num_junctions : wn.num_junctions + wn.num_tanks]) 
        self._reservoirs_data = ObjData(wn.reservoir_name_list, self._nodes_data.ids[-wn.num_reservoirs:])
        
        self._nodes = NodeRegistry(wn) 
        self._add_ids_to_nodes_and_links()
        
        self._fluid_density = wn.options.hydraulic.specific_gravity * 1000
        self._fluid_heat_capacity = wn.options.thermal.heat_capacity
        
        self._hydraulic = self._extract_relevant_hydraulic_data(hyd_sim_results)

        self._temperatures = np.zeros((self._time.stamps_num, self._nodes_data.num))
        self._tanks_volumes = np.zeros((self._time.stamps_num, self._tanks_data.num))
        self._coef_matrix = None
        self._result_vector = None
    
    def _add_ids_to_nodes_and_links(self):
        for node_name in self._nodes_data.names:
            # 1- Fill the new NodeRegistery with the model's nodes but in the same order as in the columns of results.node object
            # 2- Create new properties: _connected_links_id list and the _neighbour_nodes_id list that use ids instead of names
            self._nodes._data[node_name] = self._model.nodes[node_name]
            self._nodes[node_name]._id = self._nodes_data.name2id[node_name]
            self._nodes[node_name]._connections.connected_links_ids = np.array( [self._links_data.name2id[i] for i in self._nodes[node_name]._connections.connected_links] )
            self._nodes[node_name]._connections.neighbour_nodes_ids = np.array( [self._nodes_data.name2id[i] for i in self._nodes[node_name]._connections.neighbour_nodes] )
        
        for link_name in self._links_data.names:
            self._model.links[link_name]._id = self._links_data.name2id[link_name]
    
    def _extract_relevant_hydraulic_data(self, hyd_sim_results: SimulationResults):
        hydraulic_data = HydraulicData()
        hydraulic_data.flow_val = np.abs(hyd_sim_results.link['flowrate'].values) 
        hydraulic_data.flow_dir = np.sign(hyd_sim_results.link['flowrate'].values) 
        hydraulic_data.demands = hyd_sim_results.node['demand'].values 
        hydraulic_data.tanks_levels = hyd_sim_results.node['head'].values[:,self._tanks_data.ids] - np.array( [self._nodes[tank_name].elevation for tank_name in self._tanks_data.names] )
        return hydraulic_data
        
    def _initialize_matrices(self):
        self._temperatures[0,:-self._reservoirs_data.num] = [self._nodes[node_name]._initial_temperature for node_name in self._junctions_data.names+self._tanks_data.names]
        self._temperatures[0,-self._reservoirs_data.num:] = [self._nodes[node_name].temperature_at(0) for node_name in self._reservoirs_data.names] 
        self._tanks_volumes[0,:] = [self._nodes[tank_name].get_volume(self._nodes[tank_name].init_level) for tank_name in self._tanks_data.names]
            
    
    def _build_linear_system(self, t, node, acc_volume = 0):
        in_out = self._hydraulic.flow_dir[t, node._connections.connected_links_ids] * node._connections.connection_side
        
        upstream_links_ids = node._connections.connected_links_ids[in_out > 0] 
        upstream_nodes_ids = node._connections.neighbour_nodes_ids[in_out > 0]
        downstream_links_ids = node._connections.connected_links_ids[in_out <= 0]
        
        rho_cp_R_inv = node.total_thermal_resistance_reciprocal / (self._fluid_density * self._fluid_heat_capacity)
        
        self._coef_matrix[node._id, node._id] = (node.cell_volume + acc_volume)/ self._time.step + self._hydraulic.flow_val[t, downstream_links_ids].sum() + self._hydraulic.demands[t, node._id] + rho_cp_R_inv                                                  
        self._coef_matrix[node._id, upstream_nodes_ids] = -self._hydraulic.flow_val[t, upstream_links_ids] 
        self._result_vector[node._id] = node._soil_props.temperature_at(self._time.stamps[t]) * rho_cp_R_inv  + self._temperatures[t-1, node._id] * (node.cell_volume + acc_volume)/ self._time.step
        
        
    def run_sim(self):
        self._initialize_matrices()
        
        for t in self._time.stamps_idx[1:]:
            self._temperatures[t,-self._reservoirs_data.num:] = [self._nodes[node_name].temperature_at(self._time.stamps[t]) for node_name in self._reservoirs_data.names] 
            self._tanks_volumes[t,:] = [self._nodes[self._nodes_data.names[tank_id]].get_volume(self._hydraulic.tanks_levels[t, tank_id-self._junctions_data.num]) for tank_id in self._tanks_data.ids]
            
            self._coef_matrix = np.identity(self._nodes_data.num)
            self._result_vector = np.zeros(self._nodes_data.num)
            self._result_vector[-self._reservoirs_data.num:] = self._temperatures[t,-self._reservoirs_data.num:]
            
            for junction_name in self._junctions_data.names:
                self._build_linear_system(t, self._nodes[junction_name])
                
            for tank_name in self._tanks_data.names:
                self._build_linear_system(t, self._nodes[tank_name], self._tanks_volumes[self._time.stamps[t], self._nodes[tank_name]._id-self._junctions_data.num])

            self._temperatures[t,:] = np.linalg.solve(self._coef_matrix, self._result_vector)
        return pd.DataFrame(self._temperatures, index=self._time.stamps, columns= self._nodes_data.names)


class TimeData(object):
    """ A structure that holds time data together. Only used by the ThermalSimulator class to organize the code
        Parameters 
        ----------
        timestep : float
                   the assigned timestep for the simulation
        duration : float
                   the duration assigned for the simulation"""
    def __init__(self, timestep:float, duration:float):
        self.step = timestep
        self.duration = duration
        self.stamps = np.arange(0, duration + timestep, timestep, dtype='int64')
        self.stamps_num = len(self.stamps)
        self.stamps_idx = np.arange(self.stamps_num)


class ObjData(object):
    """A structure that holds common data for network objects together. Only used by the ThermalSimulator class to organize the code
       Parameters
       ----------
       names : a list of names of all object of specific type
       ids   : a numpy array of integers corresponding to each of the names"""
    def __init__(self, names:list[str], ids: np.array):
        self.names=names
        self.ids=ids
        self.num=len(names)
        self.name2id = {k:v for (k,v) in zip(names, ids)}

        
class HydraulicData(object):
    """A structure that holds relevant hydraulic data together. Only used by the ThermalSimulator class to organize the code"""
    def _init__(self):
        self.flow_val
        self.flow_dir
        self.demands
        self.tanks_levels


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