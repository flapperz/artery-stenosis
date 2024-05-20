import glob
import json

src_dir = '/Users/flap/Downloads/SlicerDICOMDatabase'
dst_dir = './parsed_markup'
files = glob.glob(src_dir + '/*.mrk.json')

for fpath in files:
    with open(fpath) as f:
        mrk = json.load(f)
    mrk['@schema'] = (
        'https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.0.json#'
    )
    fname = fpath[fpath.rfind('/') + 1 :]
    print(fname)
    with open(dst_dir + '/' + fname, 'w') as fp:
        json.dump(mrk, fp)
