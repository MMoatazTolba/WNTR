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
from wntr.network.elements import Weather
from wntr.sim.results import SimulationResults
from wntr.morph import split_pipe

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
        self.pipe_bc = None
        self.soil_bc = None
        self.air_bc = None
        
class HydraulicData(object):
    """A structure that holds relevant hydraulic data together. Only used by the ThermalSimulator class to organize the code"""
    def _init__(self):
        self.flow_val
        self.flow_dir
        self.demands
        self.tanks_levels
        
        
class ThermalSimulator:
    """The ThermalSimulator class employs the upwind finite-volume dicretization method to solve the energy equation in 
       Water Distribution networks. 
       
       Parameters
       ----------
       
       wn : :class:`~wntr.network.model.WaterNetworkModel`
           The water network model.
       hyd_sim_results : :class:`~wntr.network.results.SimulationResults`
           The object containing results from the hydraulic simulation.
           Only flowrates, demands, and tanks' heads are used
       weather: :class:`~wntr.network.elements.Weather`
           The object containing weather data which directly affects the network thermally"""
    
    def __init__(self, wn: WaterNetworkModel, hyd_sim_results: SimulationResults, weather: Weather = None):
        
        self._model = wn
        self._results = hyd_sim_results
        self._weather = weather
        
        self._time = TimeData(wn.options.time.hydraulic_timestep, wn.options.time.duration)
        
        self._nodes_data      = ObjData(wn.junction_name_list + wn.tank_name_list + wn.reservoir_name_list , np.arange(wn.num_nodes))
        self._links_data      = ObjData(wn.pipe_name_list + wn.head_pump_name_list + wn.power_pump_name_list + wn.valve_name_list , np.arange(wn.num_links))
        
        self._nodes_name2id   = {k:v for (k,v) in zip(self._nodes_data.names, self._nodes_data.ids)}
        self._links_name2id   = {k:v for (k,v) in zip(self._links_data.names, self._links_data.ids)}
        
        self._junctions_data  = ObjData(wn.junction_name_list, self._nodes_data.ids[0:wn.num_junctions])  
        self._tanks_data      = ObjData(wn.tank_name_list, self._nodes_data.ids[wn.num_junctions : wn.num_junctions + wn.num_tanks]) 
        self._reservoirs_data = ObjData(wn.reservoir_name_list, self._nodes_data.ids[-wn.num_reservoirs:])
        
        self._nodes = NodeRegistry(wn) 
        self._assign_ids_to_nodes_and_links()
        
        self._fluid_density = wn.options.hydraulic.specific_gravity * 1000
        self._fluid_heat_capacity = wn.options.thermal.heat_capacity
        
        self._hydraulic = self._extract_relevant_hydraulic_data()

        self._temperatures = np.zeros((self._time.stamps_num, self._nodes_data.num))
        self._soil_temperatures = np.zeros((self._time.stamps_num, self._nodes_data.num))
        self._tanks_volumes = np.zeros((self._time.stamps_num, self._tanks_data.num))
        self._coef_matrix = None
        self._result_vector = None
    
    def _assign_ids_to_nodes_and_links(self):
        for node_name in self._nodes_data.names:
            # 1- Fill the new NodeRegistery with the model's nodes but in the same order as in the columns of results.node object
            # 2- Create new properties: _connected_links_id list and the _neighbour_nodes_id list that use ids instead of names
            self._nodes._data[node_name] = self._model.nodes[node_name]
            self._nodes[node_name]._id = self._nodes_name2id[node_name]
            self._nodes[node_name]._ID = self._nodes[node_name]._id + self._nodes_data.num
            self._nodes[node_name]._connections.connected_links_ids = np.array( [self._links_name2id[i] for i in self._nodes[node_name]._connections.connected_links] )
            self._nodes[node_name]._connections.neighbour_nodes_ids = np.array( [self._nodes_name2id[i] for i in self._nodes[node_name]._connections.neighbour_nodes] )
        
        for link_name in self._links_data.names:
            self._model.links[link_name]._id = self._links_name2id[link_name]
    
    def _classify_nodes_per_boundary_condition(self, node_obj:ObjData):
        pipe_bc_nodes_names = []; soil_bc_nodes_names = []; air_bc_nodes_names = []
        pipe_bc_nodes_ids = []; soil_bc_nodes_ids = []; air_bc_nodes_ids = []
        
        for node_name in node_obj.names:
            match self._nodes[node_name]._thermal_bc:
                case 'pipe':
                    pipe_bc_nodes_names.append(node_name)
                    pipe_bc_nodes_ids.append(self._nodes[node_name]._id)
                case 'soil':
                    soil_bc_nodes_names.append(node_name)
                    soil_bc_nodes_ids.append(self._nodes[node_name]._id)
                case 'air':
                    air_bc_nodes_names.append(node_name)
                    air_bc_nodes_ids.append(self._nodes[node_name]._id)
                    
        pipe_bc = ObjData(pipe_bc_nodes_names, np.array(pipe_bc_nodes_ids, dtype=np.int32))
        soil_bc = ObjData(soil_bc_nodes_names, np.array(soil_bc_nodes_ids, dtype=np.int32))
        air_bc = ObjData(air_bc_nodes_names, np.array(air_bc_nodes_ids, dtype=np.int32))
        
        pipe_bc.IDS = pipe_bc.ids + self._nodes_data.num
        soil_bc.IDS = soil_bc.ids + self._nodes_data.num
        air_bc.IDS = air_bc.ids + self._nodes_data.num
        
        node_obj.pipe_bc = pipe_bc
        node_obj.soil_bc = soil_bc
        node_obj.air_bc = air_bc
        return node_obj
        
    def _extract_relevant_hydraulic_data(self):
        hydraulic_data = HydraulicData()
        hydraulic_data.flow_val = np.abs(self._results.link['flowrate'].values) 
        hydraulic_data.flow_dir = np.sign(self._results.link['flowrate'].values) 
        hydraulic_data.demands = self._results.node['demand'].values + self._results.node['leak_demand'].values
        hydraulic_data.tanks_levels = self._results.node['head'].values[:,self._tanks_data.ids] - np.array( [self._nodes[tank_name].elevation for tank_name in self._tanks_data.names] )
        return hydraulic_data
        
    def _initialize_matrices(self):
        self._temperatures[0,:self._junctions_data.num + self._tanks_data.num] = [self._nodes[node_name]._initial_temperature for node_name in self._junctions_data.names+self._tanks_data.names]
        self._temperatures[0,self._reservoirs_data.ids] = [self._nodes[node_name].temperature_at(0) for node_name in self._reservoirs_data.names] 
        self._tanks_volumes[0,:] = [self._nodes[tank_name].get_volume(self._nodes[tank_name].init_level) for tank_name in self._tanks_data.names]
        self._soil_temperatures[0,:]=[self._nodes[node_name]._soil_props.temperature_at(0) for node_name in self._nodes_data.names]  
    
    def _get_upstream_downstream_ids(self, t, node):
        in_out = self._hydraulic.flow_dir[t, node._connections.connected_links_ids] * node._connections.connection_side
        
        upstream_nodes_ids = node._connections.neighbour_nodes_ids[in_out > 0]
        upstream_links_ids = node._connections.connected_links_ids[in_out > 0] 
        downstream_links_ids = node._connections.connected_links_ids[in_out <= 0]
        
        return upstream_nodes_ids, upstream_links_ids, downstream_links_ids
    
    def _calculate_thermal_resistances(self, t, node, boundary_depth = 0, include_air_convection = False):
        z = node.underground_depth - boundary_depth
        k = node._soil_props.thermal_conductivity_at(self._time.stamps[t])
        hA = self._weather.convective_heat_transfer_coef_at(self._time.stamps[t]) * node.soil_air_interface_area
        R1 = node.total_thermal_resistance 
        R2 = np.arccosh(2*z/node.soil_outer_diameter)/(2* np.pi * node.soil_length * k) + include_air_convection * 1/hA 
        
        return R1, R2
    
    def _modify_water_temperature_rows(self, t, node, R1, acc_volume = 0) :
        upstream_nodes_ids, upstream_links_ids, downstream_links_ids = self._get_upstream_downstream_ids(t, node)
        
        rho_cp_R1_inv = 1 / (self._fluid_density * self._fluid_heat_capacity * R1)
        
        self._coef_matrix[node._id, node._id] = (node.water_volume + acc_volume)/ self._time.step + self._hydraulic.flow_val[t, downstream_links_ids].sum() + self._hydraulic.demands[t, node._id] + rho_cp_R1_inv                                                  
        self._coef_matrix[node._id, upstream_nodes_ids] = -self._hydraulic.flow_val[t, upstream_links_ids] 
        self._coef_matrix[node._id, node._ID] = -rho_cp_R1_inv
        self._result_vector[node._id] = self._temperatures[t-1, node._id] * (node.water_volume + acc_volume)/ self._time.step
        
    def _modify_soil_temperature_rows(self, t, node, R1, R2, boundary_temperature, radiation_term = 0) :
        self._coef_matrix[node._ID, node._ID] = node._soil_props.volumetric_heat_capacity_at(self._time.stamps[t]) * node.soil_volume/self._time.step + 1/R1 +  1/R2
        self._coef_matrix[node._ID, node._id] = -1/R1
        C = node._soil_props.volumetric_heat_capacity_at(self._time.stamps[t-1])
        T = node._soil_props.temperature_at(self._time.stamps[t-1])
        self._result_vector[node._ID]  = C*T * node.soil_volume/self._time.step + boundary_temperature /R2 + radiation_term
        a = R1
    
    def _add_row_with_pipe_bc(self, t, node, acc_volume = 0) :
        R1 = node.total_thermal_resistance
        self._modify_water_temperature_rows(t, node, R1, acc_volume)
    
    def _add_row_with_soil_bc(self, t, node, acc_volume = 0) :    
        R1, R2 = self._calculate_thermal_resistances(t, node, boundary_depth= self._weather.depth_of_soil_temperature_device)
        boundary_temperature = self._weather.soil_temperature_at(self._time.stamps[t])
        self._modify_water_temperature_rows(t, node, R1, acc_volume)
        self._modify_soil_temperature_rows(t, node, R1, R2, boundary_temperature)
        
    def _add_row_with_air_bc(self, t, node, acc_volume = 0) :
        R1, R2 = self._calculate_thermal_resistances(t, node, include_air_convection=True)
        boundary_temperature = self._weather.air_temperature_at(self._time.stamps[t])
        radiation_term = self._weather.global_solar_radiation_at(self._time.stamps[t]) * node.soil_air_interface_area * node._soil_props.absorptivity_at(self._time.stamps[t])
        self._modify_water_temperature_rows(t, node, R1, acc_volume)
        self._modify_soil_temperature_rows(t, node, R1, R2, boundary_temperature, radiation_term)
               
    def run_sim(self):
        self._initialize_matrices()
        self._junctions_data = self._classify_nodes_per_boundary_condition(self._junctions_data)
        self._tanks_data = self._classify_nodes_per_boundary_condition(self._tanks_data)
        self._reservoirs_data = self._classify_nodes_per_boundary_condition(self._reservoirs_data)
        pipe_bc_nodes_ids =  np.concatenate((self._junctions_data.pipe_bc.ids, self._tanks_data.pipe_bc.ids, self._reservoirs_data.pipe_bc.ids))
        pipe_bc_nodes_IDS = pipe_bc_nodes_ids + self._nodes_data.num
        pipe_bc_nodes_names =  self._junctions_data.pipe_bc.names + self._tanks_data.pipe_bc.names + self._reservoirs_data.pipe_bc.names
        
        for t in self._time.stamps_idx[1:]:
            self._temperatures[t,self._reservoirs_data.ids] = [self._nodes[node_name].temperature_at(self._time.stamps[t]) for node_name in self._reservoirs_data.names] 
            self._tanks_volumes[t,:] = [self._nodes[self._nodes_data.names[tank_id]].get_volume(self._hydraulic.tanks_levels[t, tank_id-self._junctions_data.num]) for tank_id in self._tanks_data.ids]
            self._soil_temperatures[t, pipe_bc_nodes_ids]=[self._nodes[node_name]._soil_props.temperature_at(self._time.stamps[t]) for node_name in pipe_bc_nodes_names]  
            
            self._coef_matrix = np.identity(self._nodes_data.num*2)
            self._result_vector = np.zeros(self._nodes_data.num*2)
            self._result_vector[self._reservoirs_data.ids] = self._temperatures[t,self._reservoirs_data.ids]
            self._result_vector[pipe_bc_nodes_IDS] = self._soil_temperatures[t,pipe_bc_nodes_ids]
            
            for junction_name in self._junctions_data.pipe_bc.names:
                self._add_row_with_pipe_bc(t, self._nodes[junction_name])
            
            for junction_name in self._junctions_data.soil_bc.names:
                self._add_row_with_soil_bc(t, self._nodes[junction_name])
                
            for junction_name in self._junctions_data.air_bc.names:
                self._add_row_with_air_bc(t, self._nodes[junction_name])
                
            for tank_name in self._tanks_data.pipe_bc.names:
                self._add_row_with_pipe_bc(t, self._nodes[tank_name], self._tanks_volumes[self._time.stamps[t], self._nodes[tank_name]._id-self._junctions_data.num])
                
            for tank_name in self._tanks_data.soil_bc.names:
                self._add_row_with_soil_bc(t, self._nodes[tank_name], self._tanks_volumes[self._time.stamps[t], self._nodes[tank_name]._id-self._junctions_data.num])
                
            for tank_name in self._tanks_data.air_bc.names:
                self._add_row_with_air_bc(t, self._nodes[tank_name], self._tanks_volumes[self._time.stamps[t], self._nodes[tank_name]._id-self._junctions_data.num])
                
            for reservoir_name in self._reservoirs_data.soil_bc.names:
                R1, R2 = self._calculate_thermal_resistances(t, self._nodes[reservoir_name], boundary_depth=self._weather.depth_of_soil_temperature_device)
                boundary_temperature = self._weather.soil_temperature_at(self._time.stamps[t])
                self._modify_soil_temperature_rows(t, self._nodes[reservoir_name], R1, R2, boundary_temperature)
                
            for reservoir_name in self._reservoirs_data.air_bc.names:
                R1, R2 = self._calculate_thermal_resistances(t, self._nodes[reservoir_name], include_air_convection=True)
                boundary_temperature = self._weather.air_temperature_at(self._time.stamps[t])
                radiation_term = self._weather.global_solar_radiation_at(self._time.stamps[t]) * self._nodes[reservoir_name].soil_area * self._nodes[reservoir_name]._soil_props.absorptivity_at(self._time.stamps[t])
                self._modify_soil_temperature_rows(t, self._nodes[reservoir_name], R1, R2, boundary_temperature, radiation_term)

            solution = np.linalg.solve(self._coef_matrix, self._result_vector)
            self._temperatures[t,:]=solution[:self._nodes_data.num]
            self._soil_temperatures[t,:]=solution[-self._nodes_data.num:]
            
        self._results.node['temperature'] = pd.DataFrame(self._temperatures, index=self._time.stamps, columns= self._nodes_data.names)
        self._results.node['soil_temperature'] = pd.DataFrame(self._soil_temperatures, index=self._time.stamps, columns= self._nodes_data.names)
        
        return self._results


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