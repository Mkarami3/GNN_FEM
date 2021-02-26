from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.data import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
dataset = Planetoid(root='/tmp/cora', name='cora')
print(len(dataset))
print(dataset.num_node_features)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# for data in loader:
#     print(data)



# mesh = meshio.read(file_name)
# print('points: ', len(mesh.points))
# print('cells: ', len(mesh.cells))
# print('point_data: ', mesh.point_data)
# print('cell_data: ', mesh.cell_data)
# print('field_data: ', mesh.field_data)
# cell_info = mesh.cells
# print(np.max(cell_info[0][1]))
# np.savetxt('test.out', cell_info[0][1], delimiter='\n', fmt='%10.1f')   # X is an array

# print(the_pts.dtype)