import argparse
import contextlib
import io
import unittest
from pathlib import Path

from experiments.synbios_moe.cli import PUBLIC_COMMANDS, build_parser


def _handler(_args: argparse.Namespace) -> None:
    pass


class SynBioSCliTest(unittest.TestCase):
    def setUp(self):
        self.root = Path("repo-root")
        self.handlers = {name: _handler for name in PUBLIC_COMMANDS}

    def test_public_command_set_is_complete(self):
        parser = build_parser(
            project_root=self.root,
            handlers=self.handlers,
            default_device="cpu",
        )

        commands = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        self.assertEqual(tuple(commands.choices), PUBLIC_COMMANDS)

    def test_probe_defaults_survive_cli_extraction(self):
        parser = build_parser(
            project_root=self.root,
            handlers=self.handlers,
            default_device="cpu",
        )

        args = parser.parse_args(
            [
                "probe",
                "--data",
                "data",
                "--model-config",
                "model.yaml",
                "--checkpoint",
                "checkpoint",
                "--attribute",
                "major",
                "--kind",
                "p",
                "--output",
                "probe.json",
            ]
        )

        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.target, "first")
        self.assertEqual(args.steps, 30_000)
        self.assertTrue(args.resume_probe)
        self.assertIs(args.func, _handler)

    def test_every_subcommand_renders_help(self):
        parser = build_parser(
            project_root=self.root,
            handlers=self.handlers,
            default_device="cpu",
        )

        for command in PUBLIC_COMMANDS:
            with self.subTest(command=command), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_status:
                    parser.parse_args([command, "--help"])
                self.assertEqual(exit_status.exception.code, 0)

    def test_handler_mismatch_is_rejected(self):
        handlers = dict(self.handlers)
        handlers.pop("prepare")

        with self.assertRaisesRegex(ValueError, "missing=.*prepare"):
            build_parser(
                project_root=self.root,
                handlers=handlers,
                default_device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
