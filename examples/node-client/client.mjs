// Drives the Python agent with the official AG-UI TypeScript client, to show
// the server speaks the protocol as published, not just our own HTML page.
//   npm install && npm start -- "weather in Tokyo?"
import { HttpAgent } from "@ag-ui/client";

const url = process.env.AGENT_URL ?? "http://localhost:8040/agent";
const question = process.argv.slice(2).join(" ") || "What is 12 * (3 + 4)? Then tell me the weather in Tenerife.";

const agent = new HttpAgent({ url });
agent.messages = [{ id: "u1", role: "user", content: question }];

let answer = "";
await agent.runAgent({}, {
  onToolCallStartEvent: ({ event }) => process.stdout.write(`\n[tool] ${event.toolCallName}(`),
  onToolCallArgsEvent: ({ event }) => process.stdout.write(event.delta),
  onToolCallEndEvent: () => process.stdout.write(")"),
  onToolCallResultEvent: ({ event }) => process.stdout.write(` -> ${event.content}\n`),
  onTextMessageContentEvent: ({ event }) => { answer += event.delta; process.stdout.write(event.delta); },
  onRunFinishedEvent: () => process.stdout.write("\n[run finished]\n"),
  onRunErrorEvent: ({ event }) => process.stdout.write(`\n[run error] ${event.message}\n`),
});
console.log("\nfinal messages:", agent.messages.length);
