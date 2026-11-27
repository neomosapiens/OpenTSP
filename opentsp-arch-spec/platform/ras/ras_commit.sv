// Independent management clock domain. Inputs have already crossed into clk.
// A matching DMA write may target only private staging storage. Publication
// additionally requires every participant's settled status and DMA visibility.
module ras_commit #(
    parameter integer CHIPS = 16,
    parameter integer TAG_BITS = 64
) (
    input wire clk, power_reset,
    input wire release_domain, isolated, health_ok, policy_permits,
    input wire launch,
    input wire [31:0] budget_ticks,
    input wire [CHIPS-1:0] members,
    input wire fault, irq_enable,
    input wire [63:0] fault_detail,
    input wire [TAG_BITS-1:0] report_tag, dma_tag, visible_tag,
    input wire [CHIPS-1:0] settled,
    input wire dma_valid, output_visible,
    output reg fenced, active,
    output reg [TAG_BITS-1:0] tag,
    output wire allow_write, publish, error_irq,
    output reg fault_latched, first_timeout, service_required,
    output reg [63:0] first_fault
);
    reg [CHIPS-1:0] participants, acknowledged;
    reg [31:0] remaining;
    reg fenced_d, active_d, fault_latched_d, first_timeout_d, service_required_d;
    reg [TAG_BITS-1:0] tag_d;
    reg [CHIPS-1:0] participants_d, acknowledged_d;
    reg [31:0] remaining_d;
    reg [63:0] first_fault_d;
    // Store parity separately from the protected state. A detected upset of
    // management state requires a cold management reset and positive isolation.
    reg control_parity;
    wire [TAG_BITS+2*CHIPS+100:0] control_word = {
        fenced, active, tag, participants, acknowledged, remaining,
        fault_latched, first_timeout, service_required, first_fault};
    wire [TAG_BITS+2*CHIPS+100:0] next_word = {
        fenced_d, active_d, tag_d, participants_d, acknowledged_d, remaining_d,
        fault_latched_d, first_timeout_d, service_required_d, first_fault_d};
    wire integrity_bad = control_parity != ^control_word;
    wire expired = active && remaining == 1;
    wire unsafe_event = fault || expired || integrity_bad;
    wire [CHIPS-1:0] arrived = report_tag == tag ? settled : {CHIPS{1'b0}};
    assign allow_write = !power_reset && !unsafe_event && !service_required && active && !fenced && dma_valid && dma_tag == tag;
    assign publish = !power_reset && !unsafe_event && !service_required && !fenced && active
                     && output_visible && visible_tag == tag && !allow_write
                     && ((acknowledged | arrived) & participants) == participants;
    assign error_irq = (fault_latched || service_required || integrity_bad) && irq_enable;

    always @* begin
        {fenced_d, active_d, tag_d, participants_d, acknowledged_d, remaining_d,
         fault_latched_d, first_timeout_d, service_required_d, first_fault_d} = control_word;
        if (power_reset) begin
            fenced_d = 1;
            active_d = 0;
            tag_d = 0;
            participants_d = 0;
            acknowledged_d = 0;
            remaining_d = 0;
            fault_latched_d = 0;
            first_fault_d = 0;
            first_timeout_d = 0;
            service_required_d = 0;
        end else if (unsafe_event || service_required) begin
            fenced_d = 1;
            active_d = 0;
            if (!fault_latched) begin
                first_fault_d = fault ? fault_detail : 0;
                first_timeout_d = expired && !fault;
            end
            fault_latched_d = 1;
            service_required_d = service_required || integrity_bad;
        end else if (fenced) begin
            if (release_domain && isolated && health_ok && policy_permits && !(&tag)) begin
                fenced_d = 0;
                fault_latched_d = 0;
            end
        end else if (active) begin
            acknowledged_d = acknowledged | arrived;
            remaining_d = remaining - 1'b1;
            if (publish) active_d = 0;
        end else if (launch && |members && |budget_ticks) begin
            if (&tag) fenced_d = 1;
            else begin
                tag_d = tag + 1'b1;
                participants_d = members;
                remaining_d = budget_ticks;
                acknowledged_d = 0;
                active_d = 1;
            end
        end
    end
    always @(posedge clk) begin
        {fenced, active, tag, participants, acknowledged, remaining,
         fault_latched, first_timeout, service_required, first_fault} <= next_word;
        control_parity <= ^next_word;
    end

`ifdef FORMAL
    reg past_valid = 0;
    always @(posedge clk) begin
        past_valid <= 1;
        if (!past_valid) assume(power_reset);
        if (past_valid) begin
            assert(!integrity_bad);
            assert(!(fenced && active));
            if (service_required) assert(fenced && !active && !publish && !allow_write);
            if (active) assert((|participants) && (|remaining));
            if (expired) assert(!publish && !allow_write);
            if (visible_tag != tag || allow_write) assert(!publish);
            if (fault || power_reset || fenced || !active) assert(!publish && !allow_write);
            if (dma_tag != tag) assert(!allow_write);
            if (publish) begin
                assert(|participants);
                assert(output_visible && ((acknowledged | arrived) & participants) == participants);
            end
            if ($past(power_reset)) assert(fenced && !active && tag == 0);
            if (!$past(power_reset)) begin
                if ($past(unsafe_event)) assert(fenced && fault_latched && !active);
                if ($past(expired && !fault && !fault_latched)) assert(first_timeout);
                if ($past(fenced) && !fenced) assert($past(release_domain && isolated && health_ok && policy_permits));
                if ($past(fault_latched && fault)) assert(first_fault == $past(first_fault));
                if ($past(fault && !fault_latched)) assert(first_fault == $past(fault_detail));
                if ($past(publish)) assert(!active);
                if (active && !$past(active)) begin
                    assert($past(launch && !fenced && !fault));
                    assert(remaining == $past(budget_ticks));
                    assert(tag == $past(tag) + 1'b1 && participants == $past(members) && acknowledged == 0);
                end
                if (active && $past(active)) begin
                    assert(remaining < $past(remaining) && remaining == $past(remaining) - 1'b1);
                    assert(participants == $past(participants));
                    assert(acknowledged == ($past(acknowledged) | ($past(report_tag == tag) ? $past(settled) : 0)));
                end
                if (tag != $past(tag)) assert(tag > $past(tag) && active && !$past(active));
                if (active && $past(active) && $past(report_tag != tag))
                    assert(acknowledged == $past(acknowledged));
            end
            cover(expired && active && !fault && !irq_enable);
            cover(active && output_visible && visible_tag != tag && !publish);
            cover(publish && |acknowledged && |(arrived & ~acknowledged));
            cover(fault && active && !irq_enable && output_visible);
            cover($past(fault_latched) && !fenced && !fault_latched);
        end
    end
`endif
endmodule
