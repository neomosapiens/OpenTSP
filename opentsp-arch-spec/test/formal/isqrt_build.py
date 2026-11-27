#!/usr/bin/env python3
"""Link an exact-root postcondition driver to receipt-pinned production C.
The parent native build establishes Sail-to-C association; this build checks
its exact current model/runtime/tool inputs and never regenerates a surrogate.
"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
import numeric_cap_build_receipt as parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    native,out=map(lambda p:Path(p).resolve(),sys.argv[1:3]);spec=Path(__file__).resolve().parents[2];model=Path(sys.argv[3]).resolve() if len(sys.argv)==4 else spec/"model";lib=Path(shutil.which("sail")).resolve().parents[1]/"share/sail/lib";out.mkdir(parents=True,exist_ok=True)
    old=json.loads((native/"build-receipt.json").read_text());current=parent.snapshot(spec,model,lib)
    if (not old.get("input_identity_equal") or old["inputs_before"]!=old["inputs_after"] or current!=old["inputs_after"] or sha(native/"model.c")!=old["generated_c_sha256"]):raise ValueError("parent actual generated C/source/tool identity mismatch")
    include_flags=["-I"+str(lib),"-I"+str(spec/"c_emulator"),"-I"+str(out)]
    link_flags=["-lgmp","-lz"]
    prefix=os.environ.get("GMP_PREFIX")
    if prefix:
        include_flags += ["-I"+str(Path(prefix)/"include"),"-I"+str(Path(prefix)/"include/x86_64-linux-gnu")]
        link_flags=[os.environ.get("GMP_LIBS",str(Path(prefix)/"lib/libgmp.so")),os.environ.get("Z_LIBS","-lz")]
    shutil.copyfile(native/"model.c",out/"model.c")
    shutil.copyfile(native/"model.h",out/"model.h")
    def snapshot():
        data=parent.snapshot(spec,model,lib)
        for n in ["isqrt_driver.c","isqrt_build.py"]:data["driver/"+n]=sha(Path(__file__).with_name(n))
        data["parent-build-receipt"]=sha(native/"build-receipt.json");data["parent-generated-c"]=sha(native/"model.c");data["parent-generated-header"]=sha(native/"model.h")
        for n in ["model.c","model.h"]:data["compiled-generated/"+n]=sha(out/n)
        for n in ["tsp_model_if.h","iq_tags.h"]:data["c-emulator/"+n]=sha(spec/"c_emulator"/n)
        # Compiler-discovered non-system include closure, not a handpicked subset.
        depcmd=["cc","-MM"]+include_flags+[str(out/"model.c"),str(Path(__file__).with_name("isqrt_driver.c")),str(spec/"test/unit_tests/stubs.c")]+[str(lib/n) for n in ["sail.c","rts.c","elf.c","sail_failure.c","sail_config.c","cJSON.c"]]
        deps=subprocess.run(depcmd,check=True,capture_output=True,text=True).stdout.replace("\\\n"," ")
        for line in deps.splitlines():
            for token in line.split(":",1)[-1].split():
                path=Path(token).resolve();data["compiler-include/"+str(path)]=sha(path)
        return data
    before=snapshot();(out/"inputs-before.json").write_text(json.dumps(before,indent=2));shutil.copyfile(native/"model.c",out/"model.c")
    cmd=["cc","-O2","-w"]+include_flags+[str(out/"model.c"),str(Path(__file__).with_name("isqrt_driver.c")),str(spec/"test/unit_tests/stubs.c")]+[str(lib/n) for n in ["sail.c","rts.c","elf.c","sail_failure.c","sail_config.c","cJSON.c"]]+["-o",str(out/"isqrt_driver.partial")]+link_flags
    subprocess.run(cmd,check=True);after=snapshot()
    if before!=after or sha(out/"model.c")!=old["generated_c_sha256"]:raise ValueError("inputs drifted while linking actual helper")
    for n in ["compiler.txt","generator.txt"]:shutil.copyfile(native/n,out/n)
    receipt={"inputs_before":before,"inputs_after":after,"input_identity_equal":True,"generated_c_sha256":sha(out/"model.c"),"generated_header_sha256":sha(out/"model.h"),"binary_sha256":sha(out/"isqrt_driver.partial"),"compiler_command":cmd,"compiler":(out/"compiler.txt").read_text(),"generator":(out/"generator.txt").read_text(),"parent_build_receipt":old,"parent_build_receipt_sha256":sha(native/"build-receipt.json"),"scope":"actual production isqrt root-floor postcondition on seven bounded argument families; not all-N Newton proof or IEEE RNE","trust":"Sail generator, C compiler, GMP/runtime/OS; not a kernel proof"}
    (out/"build-receipt.json").write_text(json.dumps(receipt,indent=2));os.replace(out/"isqrt_driver.partial",out/"isqrt_driver");print(json.dumps({"status":"BUILT","binary_sha256":receipt["binary_sha256"],"before_after_equal":True}))
if __name__=="__main__":main()
