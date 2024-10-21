agent_colors = {
    'ppo' : 'blue',
    'dqn' : 'green',
    'sac' : 'red',
    'qlearning' : 'orange',
    'sarsa' : 'purple',
    'ppo_cont' : 'blue',
    'dqn_cont' : 'green',
    'sac_cont' : 'red',
    'qlearning_cont' : 'orange',
    'sarsa_cont' : 'purple',

}


line_stype = {
    1 : '-',
    2 : '--',
    3 : ':'
}

line_type_nodes = {
    1 : '-',
    4 : '--',
    16 : ':',
    32: '--',
    63: ':',
    64 : '-.'
}

colors_nodes ={
    1: 'blue',
    4: 'green',
    16: 'red',
    32: 'orange',
    63: 'violet',
    64: 'black'

}

line_node = {
    'continuous' : '-',
    'discrete' : ':'
}

critic_model = {
    'linear' : '-',
    'network' : ':'
}

line_stype_pretrain = {
    'pretrain' : '-',
    'pretrain_layer' : '--',
    'pretrain_node' : ':',
    'pretrain_none' : '-*-',
    'pretrain_node_deterministic_node' : '-.'
}

line_type_pretrain = {
    True : '-',
    False : ':'
}

colors_partition = {
    's' : 'orange',
    'd' : 'blue',
    'ddd': 'blue',
    'sss' : 'orange',
    'sdd' : 'red',
    'dsd' : 'green',
    'dds' : 'magenta',

    'dd' : 'purple',
    'ss' : 'orange',
    'sd' : 'red',
    'ds' : 'green',
}
colors_partition_pretrain2 = {
    's' : 'red',
    'sd': 'blue',
    'ss' : 'green',
    'ds' : 'brown'
}

line_type_modelpart = {
    'backprop' : '-',
    'coagent' : ':'
}

colors_averaging ={
    1: 'blue',
    5: 'green',
    10: 'red',
    25: 'orange',
    50: 'purple',
    100: 'black'

}





'''
color
b: blue
g: green
r: red
c: cyan
m: magenta
y: yellow
k: black
w: white
'''


'''

linestyle	description
'-' or 'solid'	solid line
'--' or 'dashed'	dashed line
'-.' or 'dashdot'	dash-dotted line
':' or 'dotted'	dotted line
'None'	: draw nothing
' '	draw nothing
''	draw nothing

'''