----------------------------------------------------------------
-- enhanced_nn.lua - Evade Resolver V2 Neural Core
-- 24 inputs -> 3 outputs, 3 hidden layers of 64
----------------------------------------------------------------

local NN = {
    input_size  = 24,
    hidden_size = 64,
    hidden_layers = 3,
    output_size = 3,

    learning_rate = 0.08,
    momentum      = 0.9,
    dropout_rate  = 0.15,

    weights = {
        -- layer[i][j] = weight from neuron i (prev layer) to neuron j (this layer)
        hidden1 = {},
        hidden2 = {},
        hidden3 = {},
        output  = {},

        -- momentum buffers
        m_hidden1 = {},
        m_hidden2 = {},
        m_hidden3 = {},
        m_output  = {}
    },

    -- Prioritized experience replay
    replay = {
        buffer      = {},
        priorities  = {},
        max_size    = 4096,
        batch_size  = 32,
        min_priority = 0.05
    },

    persistence = {
        filename    = "nn_weights_latest.txt",
        save_interval = 30,  -- seconds
        last_save   = 0
    }
}

local sqrt   = math.sqrt
local exp    = math.exp
local random = math.random
local abs    = math.abs
local min    = math.min
local max    = math.max

----------------------------------------------------------------
-- Activation Functions
----------------------------------------------------------------

local function leaky_relu(x)
    return x > 0 and x or 0.01 * x
end

local function leaky_relu_deriv(x)
    return x > 0 and 1 or 0.01
end

local function sigmoid(x)
    return 1 / (1 + exp(-x))
end

local function clamp(v, mn, mx)
    if v < mn then return mn end
    if v > mx then return mx end
    return v
end

local function xavier_init(fan_in, fan_out)
    local limit = sqrt(6 / (fan_in + fan_out))
    return (random() * 2 - 1) * limit
end

----------------------------------------------------------------
-- Weight Initialization
----------------------------------------------------------------

local function init_weights_if_needed()
    -- Hidden1: input_size -> hidden_size
    if not NN.weights.hidden1[1] then
        for i = 1, NN.input_size do
            NN.weights.hidden1[i]  = {}
            NN.weights.m_hidden1[i] = {}
            for j = 1, NN.hidden_size do
                NN.weights.hidden1[i][j]  = xavier_init(NN.input_size, NN.hidden_size)
                NN.weights.m_hidden1[i][j] = 0
            end
        end
    end

    -- Hidden2 & Hidden3: hidden_size -> hidden_size
    if not NN.weights.hidden2[1] then
        for i = 1, NN.hidden_size do
            NN.weights.hidden2[i]   = {}
            NN.weights.m_hidden2[i] = {}
            NN.weights.hidden3[i]   = {}
            NN.weights.m_hidden3[i] = {}
            for j = 1, NN.hidden_size do
                NN.weights.hidden2[i][j]   = xavier_init(NN.hidden_size, NN.hidden_size)
                NN.weights.m_hidden2[i][j] = 0
                NN.weights.hidden3[i][j]   = xavier_init(NN.hidden_size, NN.hidden_size)
                NN.weights.m_hidden3[i][j] = 0
            end
        end
    end

    -- Output: hidden_size -> output_size
    if not NN.weights.output[1] then
        for i = 1, NN.hidden_size do
            NN.weights.output[i]   = {}
            NN.weights.m_output[i] = {}
            for j = 1, NN.output_size do
                NN.weights.output[i][j]   = xavier_init(NN.hidden_size, NN.output_size)
                NN.weights.m_output[i][j] = 0
            end
        end
    end
end

----------------------------------------------------------------
-- Dropout (only used during training)
----------------------------------------------------------------

local function apply_dropout(layer, rate)
    local mask = {}
    local scale = 1 / (1 - rate)
    for i = 1, #layer do
        if random() < rate then
            mask[i] = 0
            layer[i] = 0
        else
            mask[i] = scale
            layer[i] = layer[i] * scale
        end
    end
    return layer, mask
end

----------------------------------------------------------------
-- Forward Pass
-- returns: outputs, hidden_states, dropout_masks
----------------------------------------------------------------

local function forward_pass(inputs, is_training)
    init_weights_if_needed()

    -- h1
    local h1 = {}
    for j = 1, NN.hidden_size do
        local sum = 0
        for i = 1, NN.input_size do
            sum = sum + (inputs[i] or 0) * (NN.weights.hidden1[i][j] or 0)
        end
        h1[j] = leaky_relu(sum)
    end
    local mask1
    if is_training then
        h1, mask1 = apply_dropout(h1, NN.dropout_rate)
    end

    -- h2
    local h2 = {}
    for j = 1, NN.hidden_size do
        local sum = 0
        for i = 1, NN.hidden_size do
            sum = sum + h1[i] * (NN.weights.hidden2[i][j] or 0)
        end
        h2[j] = leaky_relu(sum)
    end
    local mask2
    if is_training then
        h2, mask2 = apply_dropout(h2, NN.dropout_rate)
    end

    -- h3
    local h3 = {}
    for j = 1, NN.hidden_size do
        local sum = 0
        for i = 1, NN.hidden_size do
            sum = sum + h2[i] * (NN.weights.hidden3[i][j] or 0)
        end
        h3[j] = leaky_relu(sum)
    end
    local mask3
    if is_training then
        h3, mask3 = apply_dropout(h3, NN.dropout_rate)
    end

    -- output
    local out = {}
    for j = 1, NN.output_size do
        local sum = 0
        for i = 1, NN.hidden_size do
            sum = sum + h3[i] * (NN.weights.output[i][j] or 0)
        end
        out[j] = sigmoid(sum)
    end

    return out, { h1, h2, h3 }, { mask1, mask2, mask3 }
end

----------------------------------------------------------------
-- Backward Pass (single sample)
----------------------------------------------------------------

local function backward_pass(inputs, target, outputs, hidden_states)
    local h1, h2, h3 = hidden_states[1], hidden_states[2], hidden_states[3]

    -- Output layer error
    local delta_out = {}
    for j = 1, NN.output_size do
        local y = outputs[j]
        local t = target[j] or 0
        local err = t - y
        delta_out[j] = err * y * (1 - y) -- sigmoid derivative
    end

    -- Hidden3 error
    local delta_h3 = {}
    for i = 1, NN.hidden_size do
        local sum = 0
        for j = 1, NN.output_size do
            sum = sum + delta_out[j] * (NN.weights.output[i][j] or 0)
        end
        delta_h3[i] = sum * leaky_relu_deriv(h3[i])
    end

    -- Hidden2 error
    local delta_h2 = {}
    for i = 1, NN.hidden_size do
        local sum = 0
        for j = 1, NN.hidden_size do
            sum = sum + delta_h3[j] * (NN.weights.hidden3[i][j] or 0)
        end
        delta_h2[i] = sum * leaky_relu_deriv(h2[i])
    end

    -- Hidden1 error
    local delta_h1 = {}
    for i = 1, NN.hidden_size do
        local sum = 0
        for j = 1, NN.hidden_size do
            sum = sum + delta_h2[j] * (NN.weights.hidden2[i][j] or 0)
        end
        delta_h1[i] = sum * leaky_relu_deriv(h1[i])
    end

    -- Update weights: output
    for i = 1, NN.hidden_size do
        for j = 1, NN.output_size do
            local grad = delta_out[j] * h3[i]
            local m = NN.weights.m_output[i][j] * NN.momentum + NN.learning_rate * grad
            NN.weights.m_output[i][j] = m
            NN.weights.output[i][j] = NN.weights.output[i][j] + m
        end
    end

    -- hidden3
    for i = 1, NN.hidden_size do
        for j = 1, NN.hidden_size do
            local grad = delta_h3[j] * h2[i]
            local m = NN.weights.m_hidden3[i][j] * NN.momentum + NN.learning_rate * grad
            NN.weights.m_hidden3[i][j] = m
            NN.weights.hidden3[i][j] = NN.weights.hidden3[i][j] + m
        end
    end

    -- hidden2
    for i = 1, NN.hidden_size do
        for j = 1, NN.hidden_size do
            local grad = delta_h2[j] * h1[i]
            local m = NN.weights.m_hidden2[i][j] * NN.momentum + NN.learning_rate * grad
            NN.weights.m_hidden2[i][j] = m
            NN.weights.hidden2[i][j] = NN.weights.hidden2[i][j] + m
        end
    end

    -- hidden1
    for i = 1, NN.input_size do
        for j = 1, NN.hidden_size do
            local grad = delta_h1[j] * (inputs[i] or 0)
            local m = NN.weights.m_hidden1[i][j] * NN.momentum + NN.learning_rate * grad
            NN.weights.m_hidden1[i][j] = m
            NN.weights.hidden1[i][j] = NN.weights.hidden1[i][j] + m
        end
    end
end

----------------------------------------------------------------
-- Experience Replay
----------------------------------------------------------------

local function add_experience(state, action, reward, hit_success)
    local buf   = NN.replay.buffer
    local pri   = NN.replay.priorities
    local max_n = NN.replay.max_size

    -- Priority: magnitude of reward + bonus if hit
    local p = abs(reward) + (hit_success and 0.5 or 0.0)
    p = max(p, NN.replay.min_priority)

    if #buf >= max_n then
        -- remove lowest priority
        local min_idx = 1
        for i = 2, #pri do
            if pri[i] < pri[min_idx] then
                min_idx = i
            end
        end
        table.remove(buf, min_idx)
        table.remove(pri, min_idx)
    end

    table.insert(buf, {
        state  = state,
        action = action,
        reward = reward
    })
    table.insert(pri, p)
end

local function sample_batch()
    local buf = NN.replay.buffer
    local pri = NN.replay.priorities
    local n   = #buf
    if n < NN.replay.batch_size then
        return nil
    end

    local batch = {}
    local total_p = 0
    for i = 1, n do
        total_p = total_p + pri[i]
    end

    for _ = 1, NN.replay.batch_size do
        local r = random() * total_p
        local acc = 0
        local idx = 1
        for i = 1, n do
            acc = acc + pri[i]
            if acc >= r then
                idx = i
                break
            end
        end
        table.insert(batch, buf[idx])
    end

    return batch
end

----------------------------------------------------------------
-- Training on Batch
----------------------------------------------------------------

local function train_on_batch(batch)
    if not batch or #batch == 0 then return end

    local total_loss = 0
    for i = 1, #batch do
        local expi = batch[i]
        local inputs = expi.state
        local target = expi.action

        local outputs, hidden_states = forward_pass(inputs, true)
        -- simple MSE
        local loss = 0
        for j = 1, NN.output_size do
            local err = (target[j] or 0) - (outputs[j] or 0)
            loss = loss + err * err
        end
        total_loss = total_loss + loss

        backward_pass(inputs, target, outputs, hidden_states)
    end

    -- optional LR adaptation could go here if you want
end

----------------------------------------------------------------
-- Persistence (Save / Load)
----------------------------------------------------------------

local function save_weights()
    if not writefile or not json or not globals then return end

    local now = globals.curtime()
    if now - NN.persistence.last_save < NN.persistence.save_interval then
        return
    end

    local data = {
        config = {
            input_size  = NN.input_size,
            hidden_size = NN.hidden_size,
            output_size = NN.output_size
        },
        weights = NN.weights
    }

    local ok, encoded = pcall(json.stringify, data)
    if not ok or not encoded then return end

    pcall(writefile, NN.persistence.filename, encoded)
    NN.persistence.last_save = now
end

local function load_weights()
    if not readfile or not json then return false end

    local content = readfile(NN.persistence.filename)
    if not content or content == "" then return false end

    local ok, data = pcall(json.parse, content)
    if not ok or not data or not data.weights or not data.config then
        return false
    end

    if data.config.input_size  ~= NN.input_size
    or data.config.hidden_size ~= NN.hidden_size
    or data.config.output_size ~= NN.output_size then
        return false
    end

    NN.weights = data.weights
    return true
end

----------------------------------------------------------------
-- Initialization
----------------------------------------------------------------

local function initialize()
    if not load_weights() then
        init_weights_if_needed()
    else
        -- ensure momentum buffers exist
        init_weights_if_needed()
    end
end

initialize()

----------------------------------------------------------------
-- Public Interface
----------------------------------------------------------------

return {
    NN            = NN,
    predict       = forward_pass,
    train         = train_on_batch,
    add_experience = add_experience,
    sample_batch  = sample_batch,
    save          = save_weights,
    load          = load_weights,
    initialize    = initialize
}
