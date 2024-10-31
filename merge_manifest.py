import sys
from lhotse import load_manifest, CutSet
from lhotse.utils import fastcopy


def merge_manifest(manifest_1, manifest_2, output_manifest):
    # manifest_1 is the original manifest 
    # manifest_2 is the new manifest with only custom fields
    cuts_orig = load_manifest(manifest_1)
    cuts_custom = load_manifest(manifest_2)
    
    assert len(cuts_orig) == len(cuts_custom) 
    cuts_custom = cuts_custom.sort_like(cuts_orig)
    
    new_cuts = []
    for c_orig, c_custom in zip(cuts_orig, cuts_custom):
        c_orig.custom.update(c_custom.custom)
        new_cut = fastcopy(
            c_orig,
        )
        new_cuts.append(new_cut)
        
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_jsonl(output_manifest)
    print(f"Saved to {output_manifest}")
    

if __name__=="__main__":
    manifest_1, manifest_2, output_manifest = sys.argv[1:]
    merge_manifest(
        manifest_1=manifest_1,
        manifest_2=manifest_2,
        output_manifest=output_manifest,
    )
    
