def write_xyz(types, coords, msg="", fn=None, is_onehot=False):
    # if isinstance(types, torch.Tensor):
    #     types = np.array(types.cpu())
    #     #types = np.array(types.detach().cpu())
    # if isinstance(coords, torch.Tensor):
    #     coords = np.array(coords.detach().cpu())

    # For debugging
    xyz = ""
    xyz += f"{coords.shape[0]}\n"
    xyz += msg + "\n"
    for i in range(coords.shape[0]):
        atom_type = types[i]
        xyz += f"{atom_type}\t{coords[i][0]}\t{coords[i][1]}\t{coords[i][2]}\n"
    if fn is not None:
        with open(fn, "w") as w:
            w.writelines(xyz[:-1])
    return xyz[:-1]



'''
2_sample_dimer.py의
    # 4. 스코어링 함수
    def scoring_function(points):
........
                at = Atoms(numbers=nums, positions=coords)
                valid_atoms.append(at)
                valid_indices.append(idx)
                batch_coords[idx] = coords

                여기에 아래 코드 삽입
            
'''


from makexyz import write_xyz
from rdkit import Chem

debug_dir = os.path.join(PROJECT_ROOT, 'CHECK_IF_ANGLE_MATCHES')
os.makedirs(debug_dir, exist_ok=True)

for j in range(points.shape[0]):
    at = valid_atoms[j]
    types = []
    crds = []
    for i in at: 
        sb = Chem.Atom(i.number.item()).GetSymbol()
        types.append(sb)
        crds.append(i.position)
    types = np.array(types)
    crds = np.array(crds)
    fn_name = "_".join(f"{x:.1f}" for x in points[j]) + ".xyz"
    full_path = os.path.join(debug_dir, fn_name)
    write_xyz(types, crds, fn=full_path)

print(f"Points: {len(points)}, Valid Atoms: {len(valid_atoms)}")
print(f"Debug XYZ saved to: {debug_dir}")
exit()