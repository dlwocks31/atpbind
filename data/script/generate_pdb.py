from Bio.PDB import Select, PDBIO
from Bio.PDB.PDBParser import PDBParser
from torchdrug import utils
import os
import pylcs
from torchdrug import data
import warnings

# Reference: https://stackoverflow.com/a/47584587/12134820


class ChainSelect(Select):
    def __init__(self, chain):
        self.chain = chain

    def accept_chain(self, chain):
        if chain.get_id() == self.chain:
            return 1
        else:
            return 0

    def accept_residue(self, residue):
        hetflag, _, _ = residue.get_id()
        return hetflag == " "  # Should only accept normal residue (not HETATM)

    def accept_atom(self, atom):
        # Should only accept backbone atoms
        return atom.get_name() in ["N", "CA", "C", "O"]


class LCSSelect(Select):
    def __init__(self, residue_seq):
        self.residue_seq = set(residue_seq)

    def accept_residue(self, residue):
        _, resseq, _ = residue.get_id()
        # Should only accept residue in longest substring
        return resseq in self.residue_seq


three_to_one_letter = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def read_file(path):
    '''
    Read from ATPBind dataset.
    '''
    sequences, targets, pdb_ids = [], [], []
    with open(path) as f:
        lines = f.readlines()
        num_samples = len(lines)
        for line in lines:
            sequence = line.split(' : ')[-1].strip()
            sequences.append(sequence)

            target = line.split(' : ')[-2].split(' ')
            target_indices = []
            for index in target:
                target_indices.append(int(index[1:]))
            target = []
            for index in range(len(sequence)):
                if index+1 in target_indices:
                    target.append(1)
                else:
                    target.append(0)
            targets.append(target)

            pdb_id = line.split(' : ')[0]
            pdb_ids.append(pdb_id)
    return num_samples, sequences, targets, pdb_ids


def write_parsed_pdb_from_pdb_id(atpbind_sequence, pdb_id):
    if os.path.exists("./pdb_tmp/%s.pdb" % pdb_id[:4]):
        file_path = "./pdb_tmp/%s.pdb" % pdb_id[:4]
    else:
        file_path = utils.download(
            "https://files.rcsb.org/download/%s.pdb" % pdb_id[:4], "./pdb_tmp")

    # First stage: filter for desired chain, normal residue, backbone atom
    chain_id = pdb_id[4]
    p = PDBParser(PERMISSIVE=1)
    structure = p.get_structure(pdb_id[:4], file_path)
    select = ChainSelect(chain_id)
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save('pdb_tmp/%s.pdb' % pdb_id, select)

    # Second Stage: filter for longest substring
    file_path = 'pdb_tmp/%s.pdb' % pdb_id
    p = PDBParser(PERMISSIVE=1)
    structure = p.get_structure(pdb_id, file_path)
    raw_sequence = ''.join(three_to_one_letter[residue.get_resname()]
                           for residue in structure.get_residues())
    idx_to_use = pylcs.lcs_string_idx(atpbind_sequence, raw_sequence)
    if -1 in idx_to_use:
        print('warn: -1 in lss. Using lcs instead. %s' % pdb_id)
        idx_to_use = pylcs.lcs_sequence_idx(atpbind_sequence, raw_sequence)
        assert(-1 not in idx_to_use)
    resseq_ids = [residue.get_id()[1] for i, residue in enumerate(
        structure.get_residues()) if i in idx_to_use]
    select = LCSSelect(resseq_ids)
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    pdbio.save('../pdb/%s.pdb' % pdb_id, select)


def generate_all_in_file(filename):
    _, sequences, _, pdb_ids = read_file(filename)
    for sequence, pdb_id in zip(sequences, pdb_ids):
        print('Generating %s..' % pdb_id)
        write_parsed_pdb_from_pdb_id(sequence, pdb_id)


def try_loading_pdb(file_path):
    try:
        protein = data.Protein.from_pdb(file_path)
        return protein
    except Exception as e:
        print("Error loading %s" % file_path)
        return None


def validate(base_path, filename):
    _, sequences, _, pdb_ids = read_file(os.path.join(base_path, filename))
    for sequence, pdb_id in zip(sequences, pdb_ids):
        # print('Validating %s..' % pdb_id)
        file_path = os.path.join(base_path, '%s.pdb' % pdb_id)
        protein = try_loading_pdb(file_path)
        if not protein:
            continue
        protein_sequence = ''.join(
            i for i in protein.to_sequence() if i != '.')
        if protein_sequence != sequence:
            print('validation failed for %s: sequence unmatch. len: %d %d' %
                  (pdb_id, len(protein_sequence), len(sequence)))
            print(protein_sequence)
        elif protein.num_residue != len(sequence):
            print('validation failed for %s: length unmatch. len: %d %d' %
                  (pdb_id, protein.num_residue, len(sequence)))
            continue


'''
Some generated PDB files can't be loaded using data.Protein.from_pdb because of errors
from RDKit ("Explicit valence for atom # 320 O, 3, is greater than permitted" error).
This error not only occurs when loading generated PDB files, but also when loading
the original PDB file from the PDB database.
Filtering the PDB files with `ChainSelect` resolves a few cases, but some cases 
(3CRCA, 2C7EG, 3J2TB, 3VNUA, 4QREA) still have same issues.

Also, a few generated PDB files show different residue counts when using 
data.Protein.from_pdb compared to their original sequences in ATPBind dataset. 
The generated PDB file is filtered using BioPython so that it matches the
original sequence in ATPBind dataset, so this is probably due to different implementation 
of loading PDB files in bioPython and torchprotein. (Need to investigate further)
The affected PDB files are 5J1SB, 1MABB, 3LEVH, and 3BG5A.
'''
if __name__ == '__main__':
    warnings.filterwarnings("ignore", message=".*discontinuous at line.*")
    warnings.filterwarnings("ignore", message=".*Unknown.*")
    # print('Generate train set..')
    # generate_all_in_file('../../lib/train.txt')
    # print('Generate test set..')
    # generate_all_in_file('../../lib/test.txt')

    print('Validating..')
    dataset_type = 'bosutinib'
    base_path = os.path.join(os.path.dirname(__file__), f'../../data/{dataset_type}')
    validate(base_path, f'{dataset_type}_binding.txt')
