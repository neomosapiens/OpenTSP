use super::{advance_linked_schedule_node, Error, Node, PredictedSend, Reason, Result};
use std::sync::mpsc;
use tsp_iss::ctl::PT_DATA;
use tsp_iss::harness::links::VecMsg;

struct Round {
    horizon: [u128; 4],
    first: [usize; 4],
    last: [usize; 4],
}

fn plan(now: u128, end: u128, sends: &[Vec<PredictedSend>; 4]) -> Option<Vec<Round>> {
    let mut progress = [now; 4];
    let mut first = std::array::from_fn(|i| sends[i].partition_point(|packet| packet.sent < now));
    let mut rounds = Vec::with_capacity(3);
    while progress.iter().any(|&time| time < end) {
        if rounds.len() == 3 {
            return None;
        }
        let mut horizon = [end; 4];
        for (source, packets) in sends.iter().enumerate() {
            if let Some(packet) = packets.get(first[source]) {
                if packet.target != source + 1 || packet.target >= horizon.len() {
                    return None;
                }
                horizon[packet.target] = horizon[packet.target].min(packet.ready);
            }
        }
        if horizon == progress || horizon.iter().zip(progress).any(|(&to, from)| to < from) {
            return None;
        }
        let last = std::array::from_fn(|i| sends[i].partition_point(|packet| packet.sent < horizon[i]));
        rounds.push(Round { horizon, first, last });
        progress = horizon;
        first = last;
    }
    (rounds.len() > 1).then_some(rounds)
}

pub(super) fn try_run(
    nodes: &mut [Node],
    now: u128,
    end: u128,
    window: u64,
    sends: &[Vec<PredictedSend>; 4],
    may_finish: bool,
) -> Option<Result<Reason>> {
    let rounds = plan(now, end, sends)?;
    if !overlaps_long_slices(&rounds, now, window) {
        return None;
    }
    Some(run(nodes, now, end, sends, may_finish, &rounds))
}

fn overlaps_long_slices(rounds: &[Round], now: u128, window: u64) -> bool {
    // Avoid extra worker coordination on short or uniformly busy rounds.
    // A short first slice is gated by a packet from its direct predecessor.
    // In the measured profitable geometry, its next long slice can overlap
    // another chip's long first slice that supplies no input to that consumer.
    let short = u128::from(window) * 8;
    let long = u128::from(window) * 128;
    let first = &rounds[0];
    let next = &rounds[1];
    (1..4).any(|chip| {
        let predecessor = chip - 1;
        let local = first.horizon[chip] - now;
        local > 0
            && local <= short
            && first.first[predecessor] != first.last[predecessor]
            && next.horizon[chip] - first.horizon[chip] >= long
            && (0..4).any(|other| other != chip && other != predecessor && first.horizon[other] - now >= long)
    })
}

fn run(
    nodes: &mut [Node],
    now: u128,
    end: u128,
    sends: &[Vec<PredictedSend>; 4],
    may_finish: bool,
    rounds: &[Round],
) -> Result<Reason> {
    // A chip owns its local state. At the end of each original virtual round,
    // enqueue this round's validated outgoing batch before waiting for its
    // incoming batch. Each receiver has exactly one predecessor on this route.
    let mut outgoing: [Option<mpsc::Sender<Vec<VecMsg>>>; 4] = [None, None, None, None];
    let mut incoming: [Option<mpsc::Receiver<Vec<VecMsg>>>; 4] = [None, None, None, None];
    for chip in 0..3 {
        let (tx, rx) = mpsc::channel();
        outgoing[chip] = Some(tx);
        incoming[chip + 1] = Some(rx);
    }
    std::thread::scope(|scope| -> Result<()> {
        let mut workers = Vec::with_capacity(4);
        for (chip, ((node, tx), rx)) in nodes.iter_mut().zip(outgoing).zip(incoming).enumerate() {
            workers.push(scope.spawn(move || -> Result<()> {
                let mut progress = now;
                for round in rounds {
                    let to = round.horizon[chip];
                    if progress < to {
                        advance_linked_schedule_node(node, chip, to)?;
                        progress = to;
                    }
                    let traffic = node.links.lock().unwrap().take_outbound();
                    let expected = &sends[chip][round.first[chip]..round.last[chip]];
                    if traffic.len() != expected.len() {
                        return Err(Error::Chip(format!(
                            "linked panel overlap packet count: source {chip}, expected {}, actual {}",
                            expected.len(),
                            traffic.len()
                        )));
                    }
                    for (packet, prediction) in traffic.iter().zip(expected) {
                        if packet.chip != prediction.target
                            || packet.link != prediction.link
                            || packet.arrival != prediction.ready
                            || packet.ptype != PT_DATA
                            || packet.ctl.is_some()
                            || packet.arrival < round.horizon[packet.chip]
                        {
                            return Err(Error::Chip(format!("linked panel overlap arrival violation: source {chip}")));
                        }
                    }
                    if !traffic.is_empty() {
                        // Failure here means the downstream worker already
                        // exited. Its own join preserves the original error.
                        let _ = tx.as_ref().expect("admitted sending chip").send(traffic);
                    }
                    if chip > 0 && round.last[chip - 1] != round.first[chip - 1] {
                        let packets =
                            rx.as_ref().expect("admitted receiving chip").recv().map_err(|_| {
                                Error::Chip(format!("linked panel overlap: source {} failed", chip - 1))
                            })?;
                        let mut links = node.links.lock().unwrap();
                        for packet in packets {
                            links.deliver(packet);
                        }
                    }
                }
                Ok(())
            }));
        }
        for worker in workers {
            worker.join().unwrap()?;
        }
        Ok(())
    })?;
    let next = nodes.iter().filter_map(|node| node.machine.report_time()).min();
    let reason = match next {
        Some(time) if time >= end => Reason::Limit,
        None if may_finish && nodes.iter().all(|node| node.machine.quiescent()) => Reason::Done,
        _ => return Err(Error::Chip("linked panel unexpectedly changed the public limit outcome".into())),
    };
    if nodes.iter().any(|node| node.rec.diags.len() != node.diag_seen || node.rec.irqs.len() != node.irq_seen) {
        return Err(Error::Chip("linked panel reported an unexpected interrupt or diagnostic".into()));
    }
    Ok(reason)
}
