// Command login provides interactive provider management for stackllm.
//
// Interactive mode (default):
//
//	go run ./examples/login
//
// Subcommand shortcuts:
//
//	go run ./examples/login login copilot
//	go run ./examples/login logout openai
//	go run ./examples/login status
//	go run ./examples/login models
//	go run ./examples/login default copilot/gpt-5.4
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/stack-bound/stackllm/profile"
)

func main() {
	mgr := profile.New(profile.WithCallbacks(profile.Callbacks{
		OnDeviceCode: func(userCode, verifyURL string) {
			fmt.Printf("\nOpen %s and enter code: %s\n\n", verifyURL, userCode)
		},
		OnPolling: func() {
			fmt.Print(".")
		},
		OnSuccess: func() {
			fmt.Println("\nAuthenticated!")
		},
		OnPromptKey: func(providerName string) (string, error) {
			fmt.Printf("Enter API key for %s: ", providerName)
			scanner := bufio.NewScanner(os.Stdin)
			if !scanner.Scan() {
				return "", fmt.Errorf("no input")
			}
			return strings.TrimSpace(scanner.Text()), nil
		},
		OnPromptURL: func(providerName, defaultURL string) (string, error) {
			fmt.Printf("Enter base URL for %s [%s]: ", providerName, defaultURL)
			scanner := bufio.NewScanner(os.Stdin)
			if !scanner.Scan() {
				return defaultURL, nil
			}
			v := strings.TrimSpace(scanner.Text())
			if v == "" {
				return defaultURL, nil
			}
			return v, nil
		},
	}))

	ctx := context.Background()

	// Subcommand shortcuts.
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "login":
			if len(os.Args) < 3 {
				fmt.Fprintln(os.Stderr, "Usage: login <provider>")
				os.Exit(1)
			}
			if err := mgr.Login(ctx, os.Args[2]); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			fmt.Println("Done.")
			return
		case "logout":
			if len(os.Args) < 3 {
				fmt.Fprintln(os.Stderr, "Usage: logout <provider>")
				os.Exit(1)
			}
			if err := mgr.Logout(ctx, os.Args[2]); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("Logged out of %s.\n", os.Args[2])
			return
		case "status":
			showStatus(ctx, mgr)
			return
		case "models":
			browseModels(ctx, mgr)
			return
		case "default":
			if len(os.Args) < 3 {
				fmt.Fprintln(os.Stderr, "Usage: default <provider/model>")
				os.Exit(1)
			}
			if err := mgr.SetDefault(os.Args[2]); err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				os.Exit(1)
			}
			fmt.Printf("Default set: %s\n", os.Args[2])
			return
		default:
			fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
			os.Exit(1)
		}
	}

	// Interactive menu.
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Println()
		fmt.Println("stackllm — Provider Management")
		fmt.Println()
		fmt.Println("1) Login to a provider")
		fmt.Println("2) Logout from a provider")
		fmt.Println("3) Show status")
		fmt.Println("4) Browse models")
		fmt.Println("5) Set default model")
		fmt.Println("6) Exit")
		fmt.Println()
		fmt.Print("Choose: ")

		if !scanner.Scan() {
			return
		}
		choice := strings.TrimSpace(scanner.Text())

		switch choice {
		case "1":
			menuLogin(ctx, mgr, scanner)
		case "2":
			menuLogout(ctx, mgr, scanner)
		case "3":
			showStatus(ctx, mgr)
		case "4":
			browseModels(ctx, mgr)
		case "5":
			menuSetDefault(ctx, mgr, scanner)
		case "6":
			return
		default:
			fmt.Println("Invalid choice.")
		}
	}
}

func menuLogin(ctx context.Context, mgr *profile.Manager, scanner *bufio.Scanner) {
	providers := mgr.AvailableProviders()
	fmt.Println()
	fmt.Println("Available providers:")
	for i, p := range providers {
		fmt.Printf("  %d) %s\n", i+1, p)
	}
	fmt.Println()
	fmt.Print("Choose provider: ")

	if !scanner.Scan() {
		return
	}
	idx, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	if err != nil || idx < 1 || idx > len(providers) {
		fmt.Println("Invalid choice.")
		return
	}

	providerName := providers[idx-1]
	if err := mgr.Login(ctx, providerName); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return
	}
	fmt.Printf("Logged in to %s.\n", providerName)
}

func menuLogout(ctx context.Context, mgr *profile.Manager, scanner *bufio.Scanner) {
	providers := mgr.AvailableProviders()
	fmt.Println()
	fmt.Println("Providers:")
	for i, p := range providers {
		fmt.Printf("  %d) %s\n", i+1, p)
	}
	fmt.Println()
	fmt.Print("Choose provider to logout: ")

	if !scanner.Scan() {
		return
	}
	idx, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	if err != nil || idx < 1 || idx > len(providers) {
		fmt.Println("Invalid choice.")
		return
	}

	providerName := providers[idx-1]
	if err := mgr.Logout(ctx, providerName); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return
	}
	fmt.Printf("Logged out of %s.\n", providerName)
}

func showStatus(ctx context.Context, mgr *profile.Manager) {
	statuses, err := mgr.Status(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return
	}

	fmt.Println()
	fmt.Println("Provider status:")
	for _, s := range statuses {
		marker := "  "
		if s.IsDefault {
			marker = "* "
		}
		state := "not authenticated"
		if s.Authenticated {
			state = "authenticated"
		}
		fmt.Printf("  %s%-8s  %s\n", marker, s.Name, state)
	}
}

func browseModels(ctx context.Context, mgr *profile.Manager) {
	fmt.Println()
	fmt.Println("Fetching models from authenticated providers...")

	models, err := mgr.ListAllModels(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return
	}

	if len(models) == 0 {
		fmt.Println("No models available. Login to a provider first.")
		return
	}

	fmt.Println()
	currentProvider := ""
	for i, m := range models {
		if m.Provider != currentProvider {
			currentProvider = m.Provider
			fmt.Printf("  %s:\n", currentProvider)
		}
		fmt.Printf("    %d) %s\n", i+1, m.String())
	}
	fmt.Printf("\n(%d models total)\n", len(models))
}

func menuSetDefault(ctx context.Context, mgr *profile.Manager, scanner *bufio.Scanner) {
	fmt.Println()
	fmt.Println("Fetching models from authenticated providers...")

	models, err := mgr.ListAllModels(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return
	}

	if len(models) == 0 {
		fmt.Println("No models available. Login to a provider first.")
		return
	}

	fmt.Println()
	fmt.Println("Select default model:")
	fmt.Println()
	currentProvider := ""
	for i, m := range models {
		if m.Provider != currentProvider {
			currentProvider = m.Provider
			fmt.Printf("  %s:\n", currentProvider)
		}
		fmt.Printf("    %d) %s\n", i+1, m.String())
	}
	fmt.Println()
	fmt.Print("Choose [1]: ")

	if !scanner.Scan() {
		return
	}
	input := strings.TrimSpace(scanner.Text())
	if input == "" {
		input = "1"
	}

	idx, err := strconv.Atoi(input)
	if err != nil || idx < 1 || idx > len(models) {
		fmt.Println("Invalid choice.")
		return
	}

	selected := models[idx-1]
	if err := mgr.SetDefaultModel(selected); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		return
	}
	fmt.Printf("\nDefault set: %s\n", selected.String())
}
