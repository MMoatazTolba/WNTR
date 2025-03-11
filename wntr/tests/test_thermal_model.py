import unittest
from os.path import abspath, dirname, join
from unittest import SkipTest

import numpy as np
import wntr
import wntr.network.elements as elements
from wntr.network.options import TimeOptions
from wntr.network.options import HydraulicOptions
from wntr.network.options import ThermalOptions

testdir = dirname(abspath(str(__file__)))
datadir = join(testdir, "networks_for_testing")
net1dir = join(testdir, "..", "..", "examples", "networks")


class TestThermalModel(unittest.TestCase):
    def test_node_cell_volumes(self):
        
        wn = wntr.network.WaterNetworkModel()
        wn.add_reservoir('re')
        wn.add_junction('j1')
        wn.add_junction('j2')
        wn.add_junction('j3')
        wn.add_tank('ta')
        
        wn.add_pump('pu','re','j1')
        wn.add_pipe('p1','j1','j2',length=4/np.pi*200, diameter=0.1)
        wn.add_pipe('p2','j2','ta',length=4/np.pi*400, diameter=0.1)
        wn.add_valve('va','j2','j3')
        
        self.assertEqual(np.round(wn.nodes['re'].cell_volume,3), 0.0)
        self.assertEqual(np.round(wn.nodes['j1'].cell_volume,3), 1.0)
        self.assertEqual(np.round(wn.nodes['j2'].cell_volume,3), 3.0)        
        self.assertEqual(np.round(wn.nodes['j3'].cell_volume,3), 0.0)
        self.assertEqual(np.round(wn.nodes['ta'].cell_volume,3), 2.0)
        
        wn = wntr.morph.link.reverse_link(wn, 'p2')
        
        self.assertEqual(np.round(wn.nodes['j2'].cell_volume,3), 3.0)        
        self.assertEqual(np.round(wn.nodes['ta'].cell_volume,3), 2.0)

        
        wn.links['p2'].end_node = wn.nodes['j1']
        
        self.assertEqual(np.round(wn.nodes['j1'].cell_volume,3), 3.0)
        self.assertEqual(np.round(wn.nodes['j2'].cell_volume,3), 1.0)        
        self.assertEqual(np.round(wn.nodes['ta'].cell_volume,3), 2.0)
        
        wn = wntr.morph.link.split_pipe(wn, 'p2', 'p2s', 'j_mid')
        
        self.assertEqual(np.round(wn.nodes['j1'].cell_volume,3), 2.0)      
        self.assertEqual(np.round(wn.nodes['ta'].cell_volume,3), 1.0)
        
        wn = wntr.morph.link.break_pipe(wn, 'p1', 'p1b', 'j1_break', 'j2_break')
        
        self.assertEqual(np.round(wn.nodes['j1'].cell_volume,3), 1.5)
        self.assertEqual(np.round(wn.nodes['j2'].cell_volume,3), 0.5) 
        
        wn.links['p1'].length *= 2
        
        self.assertEqual(np.round(wn.nodes['j1'].cell_volume,3), 2.0)
        
        wn.links['p1'].diameter *= 2
        
        self.assertEqual(np.round(wn.nodes['j1'].cell_volume,3), 5.0)
        
        wn.remove_link('p1b')
        
        self.assertEqual(np.round(wn.nodes['j2'].cell_volume,3), 0.0) 
        
    def test_node_thermal_resistance(self):
        
        wn = wntr.network.WaterNetworkModel()
        wn.add_reservoir('re')
        wn.add_junction('j1')
        wn.add_junction('j2')
        wn.add_junction('j3')
        wn.add_tank('ta')
        
        wn.add_pump('pu','re','j1')
        wn.add_pipe('p1','j1','j2',length=4/np.pi*200, diameter=0.1)
        wn.add_pipe('p2','j2','ta',length=4/np.pi*400, diameter=0.1)
        wn.add_valve('va','j2','j3')
        
        def half_pipe_thermal_resistance(pipe: wntr.network.Pipe):
            return (np.log(1 + 2*pipe.thickness/pipe.diameter) / pipe.thermal_conductivity + \
                    np.log(1 + 2*pipe.insulation_thickness/(pipe.diameter+2*pipe.thickness)) / pipe.insulation_thermal_conductivity) /  \
                   (np.pi * pipe.length)
        
        
        Rj1 = 1/half_pipe_thermal_resistance(wn.links['p1'])
        Rta = 1/half_pipe_thermal_resistance(wn.links['p2'])
        Rj2 = Rj1 + Rta
                                        
        self.assertAlmostEqual(wn.nodes['re'].total_thermal_resistance_reciprocal, 0.0)
        self.assertAlmostEqual(wn.nodes['j1'].total_thermal_resistance_reciprocal, Rj1)
        self.assertAlmostEqual(wn.nodes['j2'].total_thermal_resistance_reciprocal, Rj2)        
        self.assertAlmostEqual(wn.nodes['j3'].total_thermal_resistance_reciprocal, 0.0)
        self.assertAlmostEqual(wn.nodes['ta'].total_thermal_resistance_reciprocal, Rta)
        
        wn = wntr.morph.link.reverse_link(wn, 'p2')
        
        self.assertAlmostEqual(wn.nodes['j2'].total_thermal_resistance_reciprocal, Rj2)        
        self.assertAlmostEqual(wn.nodes['ta'].total_thermal_resistance_reciprocal, Rta)

        
        wn.links['p2'].end_node = wn.nodes['j1']
        
        Rj1 += Rta
        Rj2 -= Rta
        
        self.assertAlmostEqual(wn.nodes['j1'].total_thermal_resistance_reciprocal, Rj1)
        self.assertAlmostEqual(wn.nodes['j2'].total_thermal_resistance_reciprocal, Rj2)        
        self.assertAlmostEqual(wn.nodes['ta'].total_thermal_resistance_reciprocal, Rta)
        
        wn = wntr.morph.link.split_pipe(wn, 'p2', 'p2s', 'j_mid')
        
        Rj1 -= 0.5*Rta
        Rta *= 0.5
        
        self.assertAlmostEqual(wn.nodes['j1'].total_thermal_resistance_reciprocal, Rj1)      
        self.assertAlmostEqual(wn.nodes['ta'].total_thermal_resistance_reciprocal, Rta)
        
        wn = wntr.morph.link.break_pipe(wn, 'p1', 'p1b', 'j1_break', 'j2_break')
        
        Rj1 -= 0.5*Rj2
        Rj2 *= 0.5
        
        self.assertAlmostEqual(wn.nodes['j1'].total_thermal_resistance_reciprocal, Rj1)
        self.assertAlmostEqual(wn.nodes['j2'].total_thermal_resistance_reciprocal, Rj2)    
        
        wn.links['p2'].length *= 2
        wn.links['p2'].diameter = 0.05
        wn.links['p2'].thickness = 0.005
        wn.links['p2'].thermal_conductivity *= 2
        wn.links['p2'].insulation_thickness *= 1.5
        wn.links['p2'].insulation_thermal_conductivity *= 0.5
        
        
        Rta = 1/half_pipe_thermal_resistance(wn.links['p2'])
        
        self.assertAlmostEqual(wn.nodes['ta'].total_thermal_resistance_reciprocal, Rta)
        
        wn.remove_link('p1b')
        
        self.assertAlmostEqual(wn.nodes['j2'].total_thermal_resistance_reciprocal, 0.0)         
    
    def test_node_connection_data(self):
        
        wn = wntr.network.WaterNetworkModel()
        wn.add_reservoir('re')
        wn.add_junction('j1')
        wn.add_junction('j2')
        wn.add_junction('j3')
        wn.add_tank('ta')
        
        wn.add_pump('pu','re','j1')
        wn.add_pipe('p1','j1','j2')
        wn.add_pipe('p2','j2','ta')
        wn.add_valve('va','j2','j3')
        
        self.assertEqual(wn.nodes['re']._connected_links, ['pu'])
        self.assertEqual(wn.nodes['j1']._connected_links, ['pu','p1'])
        self.assertEqual(wn.nodes['j2']._connected_links, ['p1','p2','va'])        
        self.assertEqual(wn.nodes['j3']._connected_links, ['va'])
        self.assertEqual(wn.nodes['ta']._connected_links, ['p2'])
        
        self.assertEqual(wn.nodes['re']._connection_side, [-1])
        self.assertEqual(wn.nodes['j1']._connection_side, [1,-1])
        self.assertEqual(wn.nodes['j2']._connection_side, [1,-1,-1])        
        self.assertEqual(wn.nodes['j3']._connection_side, [1])
        self.assertEqual(wn.nodes['ta']._connection_side, [1])
        
        self.assertEqual(wn.nodes['re']._neighbour_nodes, ['j1'])
        self.assertEqual(wn.nodes['j1']._neighbour_nodes, ['re','j2'])
        self.assertEqual(wn.nodes['j2']._neighbour_nodes, ['j1','ta','j3'])        
        self.assertEqual(wn.nodes['j3']._neighbour_nodes, ['j2'])
        self.assertEqual(wn.nodes['ta']._neighbour_nodes, ['j2'])
        
        wn = wntr.morph.link.reverse_link(wn, 'p2')
        
        self.assertEqual(wn.nodes['j2']._connected_links, ['p1','va','p2'])        
        self.assertEqual(wn.nodes['ta']._connected_links, ['p2'])

        self.assertEqual(wn.nodes['j2']._connection_side, [1,-1,1])        
        self.assertEqual(wn.nodes['ta']._connection_side, [-1])
        
        self.assertEqual(wn.nodes['j2']._neighbour_nodes, ['j1','j3','ta'])        
        self.assertEqual(wn.nodes['ta']._neighbour_nodes, ['j2'])
        
        wn.links['p2'].end_node = wn.nodes['j1']
        
        self.assertEqual(wn.nodes['j1']._connected_links, ['pu','p1','p2'])
        self.assertEqual(wn.nodes['j2']._connected_links, ['p1','va'])        
        self.assertEqual(wn.nodes['ta']._connected_links, ['p2'])
        
        self.assertEqual(wn.nodes['j1']._connection_side, [1,-1,1])
        self.assertEqual(wn.nodes['j2']._connection_side, [1,-1])        
        self.assertEqual(wn.nodes['ta']._connection_side, [-1])
        
        self.assertEqual(wn.nodes['j1']._neighbour_nodes, ['re','j2','ta'])
        self.assertEqual(wn.nodes['j2']._neighbour_nodes, ['j1','j3'])        
        self.assertEqual(wn.nodes['ta']._neighbour_nodes, ['j1'])
        
        wn = wntr.morph.link.split_pipe(wn, 'p2', 'p2s', 'j_mid')
        
        self.assertEqual(wn.nodes['j1']._connected_links, ['pu','p1','p2s'])
        self.assertEqual(wn.nodes['ta']._connected_links, ['p2'])
        
        self.assertEqual(wn.nodes['j1']._connection_side, [1,-1,1])
        self.assertEqual(wn.nodes['ta']._connection_side, [-1])
        
        self.assertEqual(wn.nodes['j1']._neighbour_nodes, ['re','j2','j_mid'])
        self.assertEqual(wn.nodes['ta']._neighbour_nodes, ['j_mid'])
        
        wn = wntr.morph.link.break_pipe(wn, 'p1', 'p1b', 'j1_break', 'j2_break')
        
        self.assertEqual(wn.nodes['j1']._connected_links, ['pu','p2s','p1'])
        self.assertEqual(wn.nodes['j2']._connected_links, ['va','p1b'])        
        
        self.assertEqual(wn.nodes['j1']._connection_side, [1,1,-1])
        self.assertEqual(wn.nodes['j2']._connection_side, [-1,1])        
        
        self.assertEqual(wn.nodes['j1']._neighbour_nodes, ['re','j_mid','j1_break'])
        self.assertEqual(wn.nodes['j2']._neighbour_nodes, ['j3','j2_break'])        
        
        wn.remove_link('va')
        
        self.assertEqual(wn.nodes['j2']._connected_links, ['p1b'])        
        self.assertEqual(wn.nodes['j3']._connected_links, [])
        
        self.assertEqual(wn.nodes['j2']._connection_side, [1])        
        self.assertEqual(wn.nodes['j3']._connection_side, [])
        

        self.assertEqual(wn.nodes['j2']._neighbour_nodes, ['j2_break'])        
        self.assertEqual(wn.nodes['j3']._neighbour_nodes, [])
        
        
TestThermalModel().test_node_thermal_resistance()