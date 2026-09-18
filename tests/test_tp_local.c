/* In-process ("local") twin transport for DeepSeek V4.1 CUDA tensor
 * parallelism: ds4_tp_create with DS4_TP_TRANSPORT_LOCAL must pair two
 * endpoints of the SAME process through socketpairs — hello/gate-schedule
 * agreement, slab attach, row/batch/big gate exchanges, command frames,
 * ACKs and logits halves all over full-duplex in-process byte streams.
 * No GPU is involved; the slabs are host buffers exactly as the CUDA
 * engine stages them. */

#include "../ds4_tp.c"
#include <assert.h>
#include <pthread.h>
#include <time.h>

#define TEST_N_LAYER 4u
#define TEST_N_EMBD  16u

static void identity_fill(ds4_tp_identity *id) {
    memset(id, 0, sizeof(*id));
    id->gguf_bytes = 0x12345678ull;
    id->model_id = 7u;
    id->n_layer = TEST_N_LAYER;
    id->n_embd = TEST_N_EMBD;
    id->n_vocab = 1000u;
    id->quant_bits = 5u;
    id->ctx_size = 0u;   /* adopted from the leader's sessions */
    id->gate_slot_start = 0u;
    id->gate_slot_step = 1u;
    id->gates_per_token =
        TEST_N_LAYER * (uint32_t)DS4_TP_GATES_PER_LAYER;
    memset(id->gate_slot_mask, 0, sizeof(id->gate_slot_mask));
}

/* Deterministic payload for a slab slot. */
static void fill_slot(uint8_t *base, uint64_t off, uint64_t bytes,
                      uint64_t seed) {
    for (uint64_t i = 0; i < bytes; i++) {
        base[off + i] = (uint8_t)((i * 31u + seed * 53u + 11u) & 0xffu);
    }
}

static int slot_matches(const uint8_t *base, uint64_t off, uint64_t bytes,
                        uint64_t seed) {
    for (uint64_t i = 0; i < bytes; i++) {
        if (base[off + i] !=
            (uint8_t)((i * 31u + seed * 53u + 11u) & 0xffu)) {
            return 0;
        }
    }
    return 1;
}

typedef struct {
    ds4_tp *tp;
    uint8_t *slab;
    int rank;
    int rc;
    uint32_t token_seed;
} local_peer;

static void *run_rank(void *arg) {
    local_peer *p = arg;
    p->rc = 1;

    /* Lockstep decode gate sequence: two tokens, every layer, every gate.
     * The seq must match on both ranks (the protocol rejects a mismatch);
     * only the deterministic payload pat�terns differ per rank. */
    for (uint32_t tok = 0; tok < 2; tok++) {
        const uint64_t seed = 1000u + (uint64_t)p->rank * 100u + tok;
        for (uint32_t layer = 0; layer < TEST_N_LAYER; layer++) {
            for (uint32_t gate = 0; gate < DS4_TP_GATES_PER_LAYER; gate++) {
                const uint64_t out =
                    ds4_tp_slab_out_offset(p->tp, layer, gate);
                const uint64_t by = p->tp->vec_bytes;
                fill_slot(p->slab, out, by, seed);
                if (!ds4_tp_gate_exchange(p->tp, layer, gate,
                                          1000u + tok)) {
                    p->rc = 0;
                    fprintf(stderr, "rank %d: gate l=%u g=%u failed\n",
                            p->rank, layer, gate);
                    return NULL;
                }
            }
        }
        /* After the per-layer rows, each rank's in-views must hold the
         * PEER's deterministic pattern for the same token. */
        for (uint32_t layer = 0; layer < TEST_N_LAYER; layer++) {
            for (uint32_t gate = 0; gate < DS4_TP_GATES_PER_LAYER; gate++) {
                const uint64_t in = ds4_tp_slab_in_offset(p->tp, layer, gate);
                const uint64_t peer_seed =
                    1000u + (uint64_t)(1 - p->rank) * 100u + tok;
                if (!slot_matches(p->slab, in, p->tp->vec_bytes, peer_seed)) {
                    p->rc = 0;
                    fprintf(stderr, "rank %d: in-view l=%u g=%u mismatch (tok %u)\n",
                            p->rank, layer, gate, tok);
                    return NULL;
                }
            }
        }
    }

    /* Verify-block batch gate: rows x n_embd payloads for every layer. */
    for (uint32_t layer = 0; layer < TEST_N_LAYER; layer++) {
        const uint32_t rows = (layer % 2u) + 1u;
        const uint64_t out = ds4_tp_slab_batch_out_offset(p->tp, layer);
        const uint64_t by = (uint64_t)rows * TEST_N_EMBD * sizeof(float);
        fill_slot(p->slab, out, by, 500u + layer);
        if (!ds4_tp_batch_gate_exchange(p->tp, layer, rows, 77u + layer)) {
            p->rc = 0;
            fprintf(stderr, "rank %d: batch gate l=%u failed\n", p->rank, layer);
            return NULL;
        }
        const uint64_t in = ds4_tp_slab_batch_in_offset(p->tp, layer);
        if (!slot_matches(p->slab, in, by, 500u + layer)) {
            p->rc = 0;
            fprintf(stderr, "rank %d: batch in l=%u mismatch\n", p->rank, layer);
            return NULL;
        }
    }

    /* Bulk big gate: arbitrary-size symmetric payload swap. */
    {
        const uint64_t by = 234567u;
        uint8_t *out = malloc(by), *in = malloc(by);
        assert(out && in);
        fill_slot(out, 0, by, 900u + p->rank);
        if (!ds4_tp_big_gate_exchange(p->tp, 0, 31337u, out, in, by)) {
            p->rc = 0;
            fprintf(stderr, "rank %d: big gate failed\n", p->rank);
            free(out); free(in);
            return NULL;
        }
        if (!slot_matches(in, 0, by, 900u + (uint64_t)(1 - p->rank))) {
            p->rc = 0;
            fprintf(stderr, "rank %d: big gate payload mismatch\n", p->rank);
            free(out); free(in);
            return NULL;
        }
        free(out); free(in);
    }

    /* Command plane: session create + ack, eval frame, logits half. The
     * leader SENDS commands and waits for ACKs; the worker RECEIVES them
     * and acknowledges (mirroring ds4_tp_worker_run). */
    if (p->rank == 0) {
        char err[256] = "";
        assert(ds4_tp_send_session_create(p->tp, 42, 4096));
        int ack = -1;
        assert(ds4_tp_wait_command_status(p->tp, 42, &ack, "create",
                                          err, sizeof(err)));
        assert(ack == 0);

        assert(ds4_tp_send_eval(p->tp, 42, 7, 1234));
        ack = -1;
        assert(ds4_tp_wait_command_status(p->tp, 42, &ack,
                                          "eval", err, sizeof(err)));
        assert(ack == 0);

        const float half[] = {1.25f, -2.5f, 0.0f, 9.0f};
        float received[4];
        assert(ds4_tp_recv_logits_half(p->tp, received, 4));
        assert(!memcmp(received, half, sizeof(half)));
        assert(ds4_tp_send_stop(p->tp));
    } else {
        char err[256] = "";
        ds4_tp_command cmd;
        assert(ds4_tp_recv_command(p->tp, &cmd, err, sizeof(err)));
        assert(cmd.type == DS4_TP_FRAME_SESSION_CREATE &&
               cmd.session_id == 42 && cmd.value == 4096);
        ds4_tp_command_free(&cmd);
        assert(ds4_tp_send_command_ack(p->tp, 42, 0));

        assert(ds4_tp_recv_command(p->tp, &cmd, err, sizeof(err)));
        assert(cmd.type == DS4_TP_FRAME_EVAL && cmd.session_id == 42 &&
               cmd.seq == 7 && cmd.value == 1234);
        ds4_tp_command_free(&cmd);
        assert(ds4_tp_send_command_ack(p->tp, 42, 0));

        /* Worker ships its vocab-split logits half to the leader. */
        const float half[] = {1.25f, -2.5f, 0.0f, 9.0f};
        assert(ds4_tp_send_logits_half(p->tp, half, 4));
    }
    p->rc = 1;
    return NULL;
}

typedef struct {
    const ds4_tp_options *opt;
    uint8_t *slab;
    int rank;
    int rc;
} local_side;

static void *side_main(void *arg) {
    local_side *s = arg;
    char err[256] = "";
    ds4_tp_identity id;
    identity_fill(&id);
    ds4_tp *tp = NULL;
    /* ds4_tp_create runs the hello handshake: it blocks until the twin
     * endpoint answers, so both sides must create their transport from
     * separate threads (exactly as ds4_tp_local_pair_bind does). */
    if (!ds4_tp_create(&tp, s->opt, &id, err, sizeof(err))) {
        fprintf(stderr, "rank %d: create failed: %s\n", s->rank, err);
        return NULL;
    }
    if (!ds4_tp_attach_slab(tp, s->slab, err, sizeof(err))) {
        fprintf(stderr, "rank %d: slab attach failed: %s\n", s->rank, err);
        ds4_tp_free(tp);
        return NULL;
    }
    local_peer p = {.tp = tp, .slab = s->slab, .rank = s->rank};
    run_rank(&p);
    ds4_tp_detach_slab(tp);
    ds4_tp_free(tp);
    s->rc = p.rc;
    return NULL;
}

int main(void) {
    int ctl[2], data[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, ctl) == 0);
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, data) == 0);

    ds4_tp_options opt_leader = {
        .role = DS4_TP_LEADER,
        .requested = true,
        .transport = DS4_TP_TRANSPORT_LOCAL,
        .local = 1,
        .local_ctl_fd = ctl[0],
        .local_data_fd = data[0],
    };
    ds4_tp_options opt_worker = {
        .role = DS4_TP_WORKER,
        .requested = true,
        .transport = DS4_TP_TRANSPORT_LOCAL,
        .local = 1,
        .local_ctl_fd = ctl[1],
        .local_data_fd = data[1],
    };

    const uint64_t slab_bytes = ds4_tp_slab_bytes(TEST_N_LAYER, TEST_N_EMBD);
    uint8_t *slab0 = malloc(slab_bytes), *slab1 = malloc(slab_bytes);
    assert(slab0 && slab1);
    memset(slab0, 0, slab_bytes);
    memset(slab1, 0, slab_bytes);

    local_side a = {.opt = &opt_leader, .slab = slab0, .rank = 0};
    local_side b = {.opt = &opt_worker, .slab = slab1, .rank = 1};
    pthread_t thread;
    assert(pthread_create(&thread, NULL, side_main, &b) == 0);
    side_main(&a);
    assert(pthread_join(thread, NULL) == 0);
    assert(a.rc && b.rc);

    free(slab0);
    free(slab1);
    close(ctl[0]); close(ctl[1]);
    close(data[0]); close(data[1]);
    puts("TP LOCAL twin: hello/identity, row/batch/big gates, commands, "
         "acks, logits and STOP over in-process socketpairs: ok");
    return 0;
}
