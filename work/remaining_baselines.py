import subprocess,sys
for method in ['structural_replay','ewc','joint']:
 print('START_BASELINE',method,flush=True)
 subprocess.run([sys.executable,'-u','scripts/12_train_baseline.py','--method',method,'--output','outputs/pilot/'+method],check=True)
 subprocess.run([sys.executable,'scripts/09_audit_outputs.py','outputs/pilot/'+method],check=True)
 print('BASELINE_DONE',method,flush=True)
