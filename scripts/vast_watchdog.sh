#!/bin/bash
# Local watchdog for the rented box: pull results to ceph every 5 min; destroy the instance when the pipeline finished (ALL_DONE),
# when it has been dead (tmux gone) for GRACE_MIN without being restarted, or when MAX_H passes, so a dead agent session can never
# leave the box idling.
ID=$1; PORT=$2; HOST=$3; MAX_H=${4:-7}
set -a; source ~/.env; set +a
SSH="ssh -o ConnectTimeout=60 -o ServerAliveInterval=30 -p $PORT root@$HOST"
DST=/ceph/scratch/jbauer/taste; start=$(date +%s); GRACE_MIN=45; dead_since=""
pull() { rsync -az -e "ssh -p $PORT" root@$HOST:/workspace/root/ratings/ $DST/ratings/; rsync -az -e "ssh -p $PORT" root@$HOST:/workspace/root/hidden/ $DST/hidden/
         rsync -az -e "ssh -p $PORT" root@$HOST:/workspace/root/data/haiku_gen/ $DST/data/haiku_gen/; rsync -az -e "ssh -p $PORT" root@$HOST:/workspace/root/data/haiku_items.parquet $DST/data/ 2>/dev/null
         rsync -az -e "ssh -p $PORT" root@$HOST:/workspace/run.log ~/taste/logs/vast_run.log; }
while true; do
  sleep 300
  alive=$(timeout 120 $SSH 'tmux has-session -t run 2>/dev/null && echo yes || echo no' 2>/dev/null | tail -1)
  done_=$(timeout 120 $SSH 'grep -c ALL_DONE /workspace/run.log' 2>/dev/null | tail -1)
  el=$(( ($(date +%s) - start) / 3600 ))
  echo "$(date +%T) alive=$alive done=$done_ elapsed_h=$el"
  [ -z "$alive" ] && [ $el -lt $MAX_H ] && continue   # ssh hiccup: retry
  [ "$alive" = "yes" ] && { dead_since=""; pull; [ $el -lt $MAX_H ] && continue; }
  if [ "$alive" = "no" ] && [ "$done_" = "0" ] && [ $el -lt $MAX_H ]; then
    [ -z "$dead_since" ] && dead_since=$(date +%s)
    [ $(( ($(date +%s) - dead_since) / 60 )) -lt $GRACE_MIN ] && { pull; continue; }
  fi
  pull && echo "$(date +%T) pulled final results"
  echo y | vastai destroy instance $ID --api-key "$VAST_API_KEY"; echo "$(date +%T) destroyed $ID"; break
done
