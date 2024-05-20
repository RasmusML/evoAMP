import torch
import torch.nn as nn
from torch.distributions import Categorical, OneHotCategorical
from torch.nn.functional import one_hot


# from https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/mue/statearrangers.py
class Profile(nn.Module):
    """Profile HMM state arrangement.

    Parameterizes an HMM according to
    Equation S40 in [1] (with r_{M+1,j} = 1 and u_{M+1,j} = 0
    for j in {0, 1, 2}). For further background on profile HMMs see [2].

    **References**

    [1] E. N. Weinstein, D. S. Marks (2021)
    "Generative probabilistic biological sequence models that account for
    mutational variability"
    https://www.biorxiv.org/content/10.1101/2020.07.31.231381v2.full.pdf

    [2] R. Durbin, S. R. Eddy, A. Krogh, and G. Mitchison (1998)
    "Biological sequence analysis: probabilistic models of proteins and nucleic
    acids"
    Cambridge university press

    :param M: Length of regressor sequence.
    :type M: int
    :param epsilon: A small value for numerical stability.
    :type epsilon: float
    """

    def __init__(self, M, epsilon=1e-32, device=None):
        super().__init__()
        self.M = M
        self.K = 2 * M + 1
        self.epsilon = epsilon
        self.device = device

        self._make_transfer()

    def _make_transfer(self):
        """Set up linear transformations (transfer matrices) for converting from profile HMM parameters to standard HMM parameters."""
        M, K = self.M, self.K

        # Overview:
        # r -> insertion parameters
        # u -> deletion parameters
        # indices: m in {0, ..., M} and j in {0, 1, 2}; final index corresponds
        # to simplex dimensions, i.e. 1 - r and r (in that order)
        # null -> locations in the transition matrix equal to 0
        # ...transf_0 -> initial transition vector
        # ...transf -> transition matrix
        # We fix r_{M+1,j} = 1 for j in {0, 1, 2}

        self.register_buffer("r_transf_0", torch.zeros((M, 3, 2, K)))
        self.register_buffer("u_transf_0", torch.zeros((M, 3, 2, K)))
        self.register_buffer("null_transf_0", torch.zeros((K,)))

        # self.r_transf_0 = torch.zeros((M, 3, 2, K)).to(self.device)
        # self.u_transf_0 = torch.zeros((M, 3, 2, K)).to(self.device)
        # self.null_transf_0 = torch.zeros((K,)).to(self.device)

        m, g = -1, 0
        for gp in range(2):
            for mp in range(M + gp):
                kp = mg2k(mp, gp, M)
                if m + 1 - g == mp and gp == 0:
                    self.r_transf_0[m + 1 - g, g, 0, kp] = 1
                    self.u_transf_0[m + 1 - g, g, 0, kp] = 1

                elif m + 1 - g < mp and gp == 0:
                    self.r_transf_0[m + 1 - g, g, 0, kp] = 1
                    self.u_transf_0[m + 1 - g, g, 1, kp] = 1
                    for mpp in range(m + 2 - g, mp):
                        self.r_transf_0[mpp, 2, 0, kp] = 1
                        self.u_transf_0[mpp, 2, 1, kp] = 1
                    self.r_transf_0[mp, 2, 0, kp] = 1
                    self.u_transf_0[mp, 2, 0, kp] = 1

                elif m + 1 - g == mp and gp == 1:
                    if mp < M:
                        self.r_transf_0[m + 1 - g, g, 1, kp] = 1

                elif m + 1 - g < mp and gp == 1:
                    self.r_transf_0[m + 1 - g, g, 0, kp] = 1
                    self.u_transf_0[m + 1 - g, g, 1, kp] = 1
                    for mpp in range(m + 2 - g, mp):
                        self.r_transf_0[mpp, 2, 0, kp] = 1
                        self.u_transf_0[mpp, 2, 1, kp] = 1
                    if mp < M:
                        self.r_transf_0[mp, 2, 1, kp] = 1

                else:
                    self.null_transf_0[kp] = 1

        self.register_buffer("r_transf", torch.zeros((M, 3, 2, K, K)))
        self.register_buffer("u_transf", torch.zeros((M, 3, 2, K, K)))
        self.register_buffer("null_transf", torch.zeros((K, K)))

        # self.r_transf = torch.zeros((M, 3, 2, K, K)).to(self.device)
        # self.u_transf = torch.zeros((M, 3, 2, K, K)).to(self.device)
        # self.null_transf = torch.zeros((K, K)).to(self.device)

        for g in range(2):
            for m in range(M + g):
                for gp in range(2):
                    for mp in range(M + gp):
                        k, kp = mg2k(m, g, M), mg2k(mp, gp, M)
                        if m + 1 - g == mp and gp == 0:
                            self.r_transf[m + 1 - g, g, 0, k, kp] = 1
                            self.u_transf[m + 1 - g, g, 0, k, kp] = 1

                        elif m + 1 - g < mp and gp == 0:
                            self.r_transf[m + 1 - g, g, 0, k, kp] = 1
                            self.u_transf[m + 1 - g, g, 1, k, kp] = 1
                            self.r_transf[(m + 2 - g) : mp, 2, 0, k, kp] = 1
                            self.u_transf[(m + 2 - g) : mp, 2, 1, k, kp] = 1
                            self.r_transf[mp, 2, 0, k, kp] = 1
                            self.u_transf[mp, 2, 0, k, kp] = 1

                        elif m + 1 - g == mp and gp == 1:
                            if mp < M:
                                self.r_transf[m + 1 - g, g, 1, k, kp] = 1

                        elif m + 1 - g < mp and gp == 1:
                            self.r_transf[m + 1 - g, g, 0, k, kp] = 1
                            self.u_transf[m + 1 - g, g, 1, k, kp] = 1
                            self.r_transf[(m + 2 - g) : mp, 2, 0, k, kp] = 1
                            self.u_transf[(m + 2 - g) : mp, 2, 1, k, kp] = 1
                            if mp < M:
                                self.r_transf[mp, 2, 1, k, kp] = 1

                        else:
                            self.null_transf[k, kp] = 1

        self.register_buffer("vx_transf", torch.zeros((M, K)))
        self.register_buffer("vc_transf", torch.zeros((M + 1, K)))

        # self.vx_transf = torch.zeros((M, K)).to(self.device)
        # self.vc_transf = torch.zeros((M + 1, K)).to(self.device)

        for g in range(2):
            for m in range(M + g):
                k = mg2k(m, g, M)
                if g == 0:
                    self.vx_transf[m, k] = 1
                elif g == 1:
                    self.vc_transf[m, k] = 1

    def forward(
        self,
        precursor_seq_logits,
        insert_seq_logits,
        insert_logits,
        delete_logits,
        substitute_logits=None,
    ):
        """Assemble HMM parameters given profile parameters.

        :param ~torch.Tensor precursor_seq_logits: Regressor sequence
            *log(x)*. Should have rightmost dimension ``(M, D)`` and be
            broadcastable to ``(batch_size, M, D)``, where
            D is the latent alphabet size. Should be normalized to one along the
            final axis, i.e. ``precursor_seq_logits.logsumexp(-1) = zeros``.
        :param ~torch.Tensor insert_seq_logits: Insertion sequence *log(c)*.
            Should have rightmost dimension ``(M+1, D)`` and be broadcastable
            to ``(batch_size, M+1, D)``. Should be normalized
            along the final axis.
        :param ~torch.Tensor insert_logits: Insertion probabilities *log(r)*.
            Should have rightmost dimension ``(M, 3, 2)`` and be broadcastable
            to ``(batch_size, M, 3, 2)``. Should be normalized along the
            final axis.
        :param ~torch.Tensor delete_logits: Deletion probabilities *log(u)*.
            Should have rightmost dimension ``(M, 3, 2)`` and be broadcastable
            to ``(batch_size, M, 3, 2)``. Should be normalized along the
            final axis.
        :param ~torch.Tensor substitute_logits: Substitution probabilities
            *log(l)*. Should have rightmost dimension ``(D, B)``, where
            B is the alphabet size of the data, and broadcastable to
            ``(batch_size, D, B)``. Must be normalized along the
            final axis.
        :return: *initial_logits*, *transition_logits*, and
            *observation_logits*. These parameters can be used to directly
            initialize the MissingDataDiscreteHMM distribution.
        :rtype: ~torch.Tensor, ~torch.Tensor, ~torch.Tensor
        """
        initial_logits = (
            torch.einsum("...ijk,ijkl->...l", delete_logits, self.u_transf_0)
            + torch.einsum("...ijk,ijkl->...l", insert_logits, self.r_transf_0)
            + (-1 / self.epsilon) * self.null_transf_0
        )
        transition_logits = (
            torch.einsum("...ijk,ijklf->...lf", delete_logits, self.u_transf)
            + torch.einsum("...ijk,ijklf->...lf", insert_logits, self.r_transf)
            + (-1 / self.epsilon) * self.null_transf
        )
        # Broadcasting for concatenation.
        if len(precursor_seq_logits.size()) > len(insert_seq_logits.size()):
            insert_seq_logits = insert_seq_logits.unsqueeze(0).expand([precursor_seq_logits.size()[0], -1, -1])
        elif len(insert_seq_logits.size()) > len(precursor_seq_logits.size()):
            precursor_seq_logits = precursor_seq_logits.unsqueeze(0).expand([insert_seq_logits.size()[0], -1, -1])
        seq_logits = torch.cat([precursor_seq_logits, insert_seq_logits], dim=-2)

        # Option to include the substitution matrix.
        if substitute_logits is not None:
            observation_logits = torch.logsumexp(seq_logits.unsqueeze(-1) + substitute_logits.unsqueeze(-3), dim=-2)
        else:
            observation_logits = seq_logits

        return initial_logits, transition_logits, observation_logits


def mg2k(m, g, M):
    """Convert from (m, g) indexing to k indexing."""
    return m + M * g


class MuE(torch.distributions.Distribution):
    def __init__(
        self,
        precursor_seq_logits: torch.Tensor,
        insert_seq_logits: torch.Tensor,
        insert_logits: torch.Tensor,
        delete_logits: torch.Tensor,
        substitute_logits: torch.Tensor = None,
        state_arrange: Profile = None,
    ):
        if state_arrange is None:
            self.state_arrange = Profile(precursor_seq_logits.shape[-2])  # , device=precursor_seq_logits.device)
        else:
            self.state_arrange = state_arrange

        initial_logits, transition_logits, observation_logits = self.state_arrange(
            precursor_seq_logits,
            insert_seq_logits,
            insert_logits,
            delete_logits,
            substitute_logits,
        )

        self.hmm = MissingDataDiscreteHMM(initial_logits, transition_logits, observation_logits)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if value.dim() < 3:
            value = one_hot(value, self.hmm.event_shape[1]).to(torch.float32)
        return self.hmm.log_prob(value)

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return self.hmm.sample(sample_shape)


class MissingDataDiscreteHMM(torch.distributions.Distribution):
    """Hidden markov model.

    HMM with discrete latent states and discrete observations., allowing for
    missing data or variable length sequences. Observations are assumed
    to be one hot encoded; rows with all zeros indicate missing data.

    .. warning:: Unlike in pyro's pyro.distributions.DiscreteHMM, which
        computes the probability of the first state as
        initial.T @ transition @ emission
        this distribution uses the standard HMM convention,
        initial.T @ emission

    :param ~torch.Tensor initial_logits: A logits tensor for an initial
        categorical distribution over latent states. Should have rightmost
        size ``state_dim`` and be broadcastable to
        ``(batch_size, state_dim)``.
    :param ~torch.Tensor transition_logits: A logits tensor for transition
        conditional distributions between latent states. Should have rightmost
        shape ``(state_dim, state_dim)`` (old, new), and be broadcastable
        to ``(batch_size, state_dim, state_dim)``.
    :param ~torch.Tensor observation_logits: A logits tensor for observation
        distributions from latent states. Should have rightmost shape
        ``(state_dim, categorical_size)``, where ``categorical_size`` is the
        dimension of the categorical output, and be broadcastable
        to ``(batch_size, state_dim, categorical_size)``.
    """

    arg_constraints = {
        "initial_logits": torch.distributions.constraints.real,
        "transition_logits": torch.distributions.constraints.independent(torch.distributions.constraints.real, 2),
        "observation_logits": torch.distributions.constraints.independent(torch.distributions.constraints.real, 2),
    }

    def __init__(self, initial_logits, transition_logits, observation_logits, validate_args=None):
        if initial_logits.dim() < 1:
            raise ValueError(
                "expected initial_logits to have at least one dim, " f"actual shape = {initial_logits.shape}"
            )
        if transition_logits.dim() < 2:
            raise ValueError(
                "expected transition_logits to have at least two dims, " f"actual shape = {transition_logits.shape}"
            )
        if observation_logits.dim() < 2:
            raise ValueError(
                "expected observation_logits to have at least two dims, " f"actual shape = {transition_logits.shape}"
            )
        shape = _broadcast_shape(
            initial_logits.shape[:-1],
            transition_logits.shape[:-2],
            observation_logits.shape[:-2],
        )
        if len(shape) == 0:
            shape = torch.Size([1])
        batch_shape = shape
        event_shape = (1, observation_logits.shape[-1])
        self.initial_logits = initial_logits - initial_logits.logsumexp(-1, True)
        self.transition_logits = transition_logits - transition_logits.logsumexp(-1, True)
        self.observation_logits = observation_logits - observation_logits.logsumexp(-1, True)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, value):
        """Log probability of the observation sequence.

        :param ~torch.Tensor value: One-hot encoded observation. Must be
            real-valued (float) and broadcastable to
            ``(batch_size, num_steps, categorical_size)`` where
            ``categorical_size`` is the dimension of the categorical output.
            Missing data is represented by zeros, i.e.
            ``value[batch, step, :] == tensor([0, ..., 0])``.
            Variable length observation sequences can be handled by padding
            the sequence with zeros at the end.
        """
        assert value.shape[-1] == self.event_shape[1]

        # Combine observation and transition factors.
        value_logits = torch.matmul(value, torch.transpose(self.observation_logits, -2, -1))
        result = self.transition_logits.unsqueeze(-3) + value_logits[..., 1:, None, :]

        # Eliminate time dimension.
        result = _sequential_logmatmulexp(result)

        # Combine initial factor.
        result = self.initial_logits + value_logits[..., 0, :] + result.logsumexp(-1)

        # Marginalize out final state.
        result = result.logsumexp(-1)
        return result

    def sample(self, sample_shape=torch.Size([])):
        """Sample from distribution.

        :param ~torch.Size sample_shape: Sample shape, last dimension must be
            ``num_steps`` and must be broadcastable to
            ``(batch_size, num_steps)``. batch_size must be int not tuple.
        """
        # shape: batch_size x num_steps x categorical_size
        shape = _broadcast_shape(
            torch.Size(list(self.batch_shape) + [1, 1]),
            torch.Size(list(sample_shape) + [1]),
            torch.Size((1, 1, self.event_shape[-1])),
        )
        # state: batch_size x state_dim
        state = OneHotCategorical(logits=self.initial_logits).sample()
        # sample: batch_size x num_steps x categorical_size
        sample = torch.zeros(shape)
        for i in range(shape[-2]):
            # batch_size x 1 x state_dim @
            # batch_size x state_dim x categorical_size
            obs_logits = torch.matmul(state.unsqueeze(-2), self.observation_logits).squeeze(-2)
            sample[:, i, :] = OneHotCategorical(logits=obs_logits).sample()
            # batch_size x 1 x state_dim @
            # batch_size x state_dim x state_dim
            trans_logits = torch.matmul(state.unsqueeze(-2), self.transition_logits).squeeze(-2)
            state = OneHotCategorical(logits=trans_logits).sample()

        return sample

    def filter(self, value):
        """Compute the marginal probability of the state variable at each step conditional on the previous observations.

        :param ~torch.Tensor value: One-hot encoded observation.
            Must be real-valued (float) and broadcastable to
            ``(batch_size, num_steps, categorical_size)`` where
            ``categorical_size`` is the dimension of the categorical output.
        """
        # batch_size x num_steps x state_dim
        shape = _broadcast_shape(
            torch.Size(list(self.batch_shape) + [1, 1]),
            torch.Size(list(value.shape[:-1]) + [1]),
            torch.Size((1, 1, self.initial_logits.shape[-1])),
        )
        filter = torch.zeros(shape)

        # Combine observation and transition factors.
        # batch_size x num_steps x state_dim
        value_logits = torch.matmul(value, torch.transpose(self.observation_logits, -2, -1))
        # batch_size x num_steps-1 x state_dim x state_dim
        result = self.transition_logits.unsqueeze(-3) + value_logits[..., 1:, None, :]

        # Forward pass. (This could be parallelized using the
        # Sarkka & Garcia-Fernandez method.)
        filter[..., 0, :] = self.initial_logits + value_logits[..., 0, :]
        filter[..., 0, :] = filter[..., 0, :] - torch.logsumexp(filter[..., 0, :], -1, True)
        for i in range(1, shape[-2]):
            filter[..., i, :] = torch.logsumexp(filter[..., i - 1, :, None] + result[..., i - 1, :, :], -2)
            filter[..., i, :] = filter[..., i, :] - torch.logsumexp(filter[..., i, :], -1, True)
        return filter

    def smooth(self, value):
        """Compute posterior expected value of state at each position (smoothing).

        :param ~torch.Tensor value: One-hot encoded observation.
            Must be real-valued (float) and broadcastable to
            ``(batch_size, num_steps, categorical_size)`` where
            ``categorical_size`` is the dimension of the categorical output.
        """
        # Compute filter and initialize.
        filter = self.filter(value)
        shape = filter.shape
        backfilter = torch.zeros(shape)

        # Combine observation and transition factors.
        # batch_size x num_steps x state_dim
        value_logits = torch.matmul(value, torch.transpose(self.observation_logits, -2, -1))
        # batch_size x num_steps-1 x state_dim x state_dim
        result = self.transition_logits.unsqueeze(-3) + value_logits[..., 1:, None, :]
        # Construct backwards filter.
        for i in range(shape[-2] - 1, 0, -1):
            backfilter[..., i - 1, :] = torch.logsumexp(backfilter[..., i, None, :] + result[..., i - 1, :, :], -1)

        # Compute smoothed version.
        smooth = filter + backfilter
        smooth = smooth - torch.logsumexp(smooth, -1, True)
        return smooth

    def sample_states(self, value):
        """Sample states with forward filtering-backward sampling algorithm.

        :param ~torch.Tensor value: One-hot encoded observation.
            Must be real-valued (float) and broadcastable to
            ``(batch_size, num_steps, categorical_size)`` where
            ``categorical_size`` is the dimension of the categorical output.
        """
        filter = self.filter(value)
        shape = filter.shape
        joint = filter.unsqueeze(-1) + self.transition_logits.unsqueeze(-3)
        states = torch.zeros(shape[:-1], dtype=torch.long)
        states[..., -1] = Categorical(logits=filter[..., -1, :]).sample()
        for i in range(shape[-2] - 1, 0, -1):
            logits = torch.gather(
                joint[..., i - 1, :, :],
                -1,
                states[..., i, None, None] * torch.ones([shape[-1], 1], dtype=torch.long),
            ).squeeze(-1)
            states[..., i - 1] = Categorical(logits=logits).sample()
        return states

    def map_states(self, value):
        """Compute maximum a posteriori (MAP) estimate of state variable with Viterbi algorithm.

        :param ~torch.Tensor value: One-hot encoded observation.
            Must be real-valued (float) and broadcastable to
            ``(batch_size, num_steps, categorical_size)`` where
            ``categorical_size`` is the dimension of the categorical output.
        """
        # Setup for Viterbi.
        # batch_size x num_steps x state_dim
        shape = _broadcast_shape(
            torch.Size(list(self.batch_shape) + [1, 1]),
            torch.Size(list(value.shape[:-1]) + [1]),
            torch.Size((1, 1, self.initial_logits.shape[-1])),
        )
        state_logits = torch.zeros(shape)
        state_traceback = torch.zeros(shape, dtype=torch.long)

        # Combine observation and transition factors.
        # batch_size x num_steps x state_dim
        value_logits = torch.matmul(value, torch.transpose(self.observation_logits, -2, -1))
        # batch_size x num_steps-1 x state_dim x state_dim
        result = self.transition_logits.unsqueeze(-3) + value_logits[..., 1:, None, :]

        # Forward pass.
        state_logits[..., 0, :] = self.initial_logits + value_logits[..., 0, :]
        for i in range(1, shape[-2]):
            transit_weights = state_logits[..., i - 1, :, None] + result[..., i - 1, :, :]
            state_logits[..., i, :], state_traceback[..., i, :] = torch.max(transit_weights, -2)
        # Traceback.
        map_states = torch.zeros(shape[:-1], dtype=torch.long)
        map_states[..., -1] = torch.argmax(state_logits[..., -1, :], -1)
        for i in range(shape[-2] - 1, 0, -1):
            map_states[..., i - 1] = torch.gather(
                state_traceback[..., i, :], -1, map_states[..., i].unsqueeze(-1)
            ).squeeze(-1)
        return map_states

    def given_states(self, states):
        """Distribution conditional on the state variable.

        :param ~torch.Tensor map_states: State trajectory. Must be
            integer-valued (long) and broadcastable to
            ``(batch_size, num_steps)``.
        """
        shape = _broadcast_shape(
            list(self.batch_shape) + [1, 1],
            list(states.shape[:-1]) + [1, 1],
            [1, 1, self.observation_logits.shape[-1]],
        )
        states_index = states.unsqueeze(-1) * torch.ones(shape, dtype=torch.long)
        obs_logits = self.observation_logits * torch.ones(shape)
        logits = torch.gather(obs_logits, -2, states_index)
        return OneHotCategorical(logits=logits)

    def sample_given_states(self, states):
        """Sample an observation conditional on the state variable.

        :param ~torch.Tensor map_states: State trajectory. Must be
            integer-valued (long) and broadcastable to
            ``(batch_size, num_steps)``.
        """
        conditional = self.given_states(states)
        return conditional.sample()


class _SafeLog(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.log()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / x.clamp(min=torch.finfo(x.dtype).eps)


def _logmatmulexp(x, y):
    """Numerically stable version of ``(x.exp() @ y.exp()).log()``."""
    finfo = torch.finfo(x.dtype)  # avoid nan due to -inf - -inf
    x_shift = x.detach().max(-1, keepdim=True).values.clamp_(min=finfo.min)
    y_shift = y.detach().max(-2, keepdim=True).values.clamp_(min=finfo.min)
    xy = safe_log(torch.matmul((x - x_shift).exp(), (y - y_shift).exp()))
    return xy + x_shift + y_shift


def safe_log(x):
    """Like :func:`torch.log` but avoids infinite gradients at log(0) by clamping them to at most ``1 / finfo.eps``."""
    return _SafeLog.apply(x)


# TODO re-enable jitting once _SafeLog is supported by the jit.
# See https://discuss.pytorch.org/t/does-torch-jit-script-support-custom-operators/65759/4
# @torch_jit_script_if_tracing
def _sequential_logmatmulexp(logits):
    """Sequentially computes the log of the matrix product of a sequence of tensors.

    For a tensor ``x`` whose time dimension is -3, computes::

        x[..., 0, :, :] @ x[..., 1, :, :] @ ... @ x[..., T-1, :, :]

    but does so numerically stably in log space.
    """
    batch_shape = logits.shape[:-3]
    state_dim = logits.size(-1)
    while logits.size(-3) > 1:
        time = logits.size(-3)
        even_time = time // 2 * 2
        even_part = logits[..., :even_time, :, :]
        x_y = even_part.reshape(batch_shape + (even_time // 2, 2, state_dim, state_dim))
        x, y = x_y.unbind(-3)
        contracted = _logmatmulexp(x, y)
        if time > even_time:
            contracted = torch.cat((contracted, logits[..., -1:, :, :]), dim=-3)
        logits = contracted
    return logits.squeeze(-3)


def _broadcast_shape(*shapes, **kwargs):
    """Similar to ``np.broadcast()`` but for shapes.

    Equivalent to ``np.broadcast(*map(np.empty, shapes)).shape``.

    :param tuple shapes: shapes of tensors.
    :param bool strict: whether to use extend-but-not-resize broadcasting.
    :returns: broadcasted shape
    :rtype: tuple
    :raises: ValueError
    """
    strict = kwargs.pop("strict", False)
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError(
                    "shape mismatch: objects cannot be broadcast to a single shape: {}".format(
                        " vs ".join(map(str, shapes))
                    )
                )
    return tuple(reversed(reversed_shape))
