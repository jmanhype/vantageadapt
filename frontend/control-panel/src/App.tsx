import { AppShell, Container, Stack } from "@mantine/core";
import SystemStatus from "./components/SystemStatus";
import StrategyForm from "./components/StrategyForm";
import StrategyList from "./components/StrategyList";

export default function App() {
  return (
    <AppShell>
      <Container size="lg" py="xl">
        <Stack gap="xl">
          <SystemStatus />
          <StrategyForm />
          <StrategyList />
        </Stack>
      </Container>
    </AppShell>
  );
}
