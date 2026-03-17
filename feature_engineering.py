import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_fingerprint(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(2048)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius=2,
        nBits=2048
    )

    return np.array(fp)